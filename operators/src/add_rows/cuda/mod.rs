use super::{AddRows, Args};
use crate::{
    add_rows::args::Meta,
    cuda::{Gpu, Handle, ModuleBox},
    get_static,
    utils::gcd,
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};
use std::{
    ffi::{c_uint, CString},
    sync::Arc,
};

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
        let Meta {
            batch: b,
            m,
            n,
            k,
            ..
        } = args.meta()?;
        let Args {
            dst_layout,
            dst_base,
            src_layout,
            src_base,
            idx_layout,
            idx_base,
        } = args;

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
            b   m   n   k
            bsd msd nsd
            bsi msi nss kss
        }
        // todo
        let params =
            cuda::params![dst_base, src_base, idx_base,b,m, n, k, bsd, msd, nsd, kss, nss, bsi, msi];
        let block = gcd(self.max_threads_block, 1);
        self.module.launch(
            CString::new(NAME).unwrap(),
            1 as c_uint,
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

extern "C" __global__ void  {NAME}(
    half *__restrict__ c,
    int const * __restrict__ c_strides,
    half  const *__restrict__ a,
    int const *__restrict__ a_strides,
    half const *__restrict__ b,
    int const *__restrict__ b_strides,
    int const count,
    int const *__restrict__ i_strides,
    int const i_strides_size) {{
    add(c, c_strides, a, a_strides, b, b_strides, count, i_strides, i_strides_size);
    }}"#
    )
}

#[cfg(test)]
mod test {
    use std::ptr::null;

    use super::{Args, Gpu, Operator};
    use crate::{dyn_, Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F64},
        DigitLayout,
    };

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
        h: usize,
        w: usize,
        d_base: *mut H::Byte,
        s_base: *const H::Byte,
        i_base: *const H::Byte,
    ) -> Args<H> {
        Args {
            dst_layout: TensorLayout::new_contiguous(dt, &[b,h,w]),
            dst_base: d_base,
            src_layout: TensorLayout::new_contiguous(dt, &[h,w]),
            src_base: s_base,
            idx_layout: TensorLayout::new_contiguous(dt, &[h,w]),
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
            cuda::cast_load,
            test_utils::{Diff, ErrorCollector},
        };
        use cuda::memcpy_d2h;
        use half::f16;
        use rand::Rng;

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);
        cpu_op.scheme(&dyn_args(F64), 0).unwrap();
        gpu_op.scheme(&dyn_args(F16), 0).unwrap();

        let n = 1;
        let h = 1024;
        let w = 1024;
        let len = n * h * w;
        let mut c = vec![0.0f64; len];
        let mut a = vec![0.1f64; len];
        let mut b = vec![0.1f64; len];
        rand::thread_rng().fill(&mut a[..]);
        rand::thread_rng().fill(&mut b[..]);
        let data_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            let mut c = cast_load(&c, f16::from_f64, &stream);
            let a = cast_load(&a, f16::from_f64, &stream);
            let b = cast_load(&b, f16::from_f64, &stream);
            gpu_op
                .launch(
                    &args(
                        F16,
                        n,
                        h,
                        w,
                        c.as_mut_ptr().cast(),
                        a.as_ptr().cast(),
                        b.as_ptr().cast(),
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![f16::ZERO; len];
            memcpy_d2h(&mut host, &c);
            host
        });
        cpu_op
            .launch(
                &args(
                    F64,
                    n,
                    h,
                    w,
                    c.as_mut_ptr().cast(),
                    a.as_ptr().cast(),
                    b.as_ptr().cast(),
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();
        let diff = c
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
