use super::{AddRows, Args};
use crate::{
    add_rows::args::Meta,
    cuda::{Gpu, Handle, ModuleBox},
    get_static, strides_not_support, type_not_support,
    utils::gcd,
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use digit_layout::types::F16;
use std::{ffi::CString, sync::Arc};

pub struct Operator {
    _handle: Arc<Handle>,
    max_threads_block: usize,
    module: Arc<ModuleBox>,
}

const NAME: &str = "add_rows_f16";
const CODE: &str = include_str!("add_rows.cuh");
impl AddRows<Gpu> for Operator {}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        let device = node.0.device();
        Self {
            _handle: node.0.clone(),
            max_threads_block: device.block_limit().max_threads,
            module: node
                .0
                .compile_kernel(NAME, device.compute_capability(), format_code),
        }
    }

    #[inline]
    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        Ok(0)
    }

    fn launch<QA>(
        &self,
        args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let Meta { batch: b, n, m, .. } = args.meta()?;

        let Args {
            dst_layout,
            dst_base,
            src_layout,
            src_base,
            idx_layout,
            idx_base,
        } = args;
        if dst_layout.dt() != F16 {
            Err(type_not_support(" add_rows currently only supports f16"))?
        }
        let &[bsd, msd, nsd] = dst_layout.strides() else {
            unreachable!()
        };
        let &[kss, nss] = src_layout.strides() else {
            unreachable!()
        };
        let &[bsi, msi] = idx_layout.strides() else {
            unreachable!()
        };

        get_static! {
            b   n   m
            bsd msd nsd
            bsi msi nss kss
        }
        let unit_dst = dst_layout.dt().nbytes() as isize;
        let unit_idx = idx_layout.dt().nbytes() as isize;
        if nsd != unit_dst || nss != unit_dst || msi != unit_idx {
            return Err(strides_not_support("").into());
        };
        fn cast(strides: &[isize], size: usize) -> Vec<isize> {
            strides.iter().map(|x| x / size as isize).collect()
        }
        let &[bsd, msd, kss] = cast(&[bsd, msd, kss], unit_dst as usize).as_slice() else {
            todo!()
        };
        let &[bsi] = cast(&[bsi], unit_idx as usize).as_slice() else {
            todo!()
        };
        let params = cuda::params![dst_base, src_base, idx_base, bsd, msd, kss, bsi];
        let block = gcd(self.max_threads_block, n);
        let dimx = n.div_ceil(block);
        self.module.launch(
            CString::new(NAME).unwrap(),
            (b as _, m as _, dimx as _),
            block as u32,
            params.as_ptr(),
            0,
            queue_alloc.queue(),
        );
        Ok(())
    }
}

fn format_code() -> String {
    format!(
        r#"{CODE}
extern "C" __global__ void {NAME}(
        half *__restrict__ dst,
        half const *__restrict__ src,
        unsigned int const *__restrict__ idx,
        int const stride_d_b,
        int const stride_d_m,
        int const stride_s,
        int const stride_i){{
        add_rows(dst, src, idx,stride_d_b, stride_d_m, stride_s, stride_i);
        }}"#
    )
}

#[cfg(test)]
mod test {
    use super::{Args, Gpu, Operator};
    use crate::{cuda::cast_load, dyn_, Hardware, Operator as _, TensorLayout};
    use cuda::memcpy_d2h;
    use digit_layout::{
        types::{F16, F64, U32},
        DigitLayout,
    };
    use half::f16;
    use std::ptr::null;

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use std::ptr::null_mut;
        let layout = TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]);
        Args {
            dst_layout: layout.clone(),
            dst_base: null_mut(),
            src_layout: layout.clone(),
            src_base: null(),
            idx_layout: layout.clone(),
            idx_base: null(),
        }
    }
    fn args<H: Hardware>(
        dt: DigitLayout,
        b: usize,
        m: usize,
        n: usize,
        k: usize,
        d_base: *mut H::Byte,
        s_base: *const H::Byte,
        i_base: *const H::Byte,
    ) -> Args<H> {
        Args {
            dst_layout: TensorLayout::new_contiguous(dt, &[b, m, n]),
            dst_base: d_base,
            src_layout: TensorLayout::new_contiguous(dt, &[k, n]),
            src_base: s_base,
            idx_layout: TensorLayout::new_contiguous(U32, &[b, m]),
            idx_base: i_base,
        }
    }
    #[test]
    fn test_compile() {
        use super::NAME;
        use std::ffi::CString;

        let Some(gpu) = Gpu::init() else {
            return;
        };
        println!("{}", gpu.0.device().info());

        let mut op = Operator::new(&gpu);
        op.scheme(&dyn_args(F16), 0).unwrap();

        gpu.apply(|ctx| {
            println!(
                "{NAME}\n{}",
                op.module.load(CString::new(NAME).unwrap(), ctx).info()
            );
        })
    }

    #[test]
    fn test_compute() {
        use super::super::common_cpu::Operator as RefOp;
        use crate::{
            common_cpu::{Cpu, ThisThread},
            test_utils::{Diff, ErrorCollector},
        };
        use rand::Rng;

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        cpu_op.scheme(&dyn_args(F64), 0).unwrap();
        gpu_op.scheme(&dyn_args(F16), 0).unwrap();

        let b = 1;
        let m = 10;
        let n = 2048;
        let k = m;
        let mut d = vec![0.1f64; b * m * n];
        let mut s = vec![0.1f64; k * n];
        let i: Vec<u32> = (0..=m).cycle().take(m * b).map(|x| x as u32).collect(); // 收集结果到 Vec 中
        rand::thread_rng().fill(&mut d[..]);
        rand::thread_rng().fill(&mut s[..]);
        let data_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            let mut d = cast_load(&d, f16::from_f64, &stream);
            let s = cast_load(&s, f16::from_f64, &stream);
            let i = cast_load(&i, u32::from, &stream);
            gpu_op
                .launch(
                    &args(
                        F16,
                        b,
                        m,
                        n,
                        k,
                        d.as_mut_ptr().cast(),
                        s.as_ptr().cast(),
                        i.as_ptr().cast(),
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![f16::ZERO; b * m * n];
            memcpy_d2h(&mut host, &d);
            host
        });
        cpu_op
            .launch(
                &args(
                    F64,
                    b,
                    m,
                    n,
                    k,
                    d.as_mut_ptr().cast(),
                    s.as_ptr().cast(),
                    i.as_ptr().cast(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();
        let diff = d
            .into_iter()
            .zip(data_ans)
            .map(|(a, b)| Diff::new(a, b.to_f64()))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
