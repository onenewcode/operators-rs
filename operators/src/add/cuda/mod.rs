use super::{args::Scheme, Add, Args};
use crate::{
    cuda::{Gpu, Handle, ModuleBox},
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

const NAME: &str = "add_f16";
const CODE: &str = include_str!("add.cuh");
impl Add<Gpu> for Operator {}

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
        let size = args.c_layout.dt().nbytes() as isize;
        let cast = |strides: &[isize]| -> Vec<isize> { strides.iter().map(|x| x / size).collect() };

        let scheme = Scheme::new(args)?;
        let strids_size = scheme.idx_strides().len() as i32;
        let idx_strides = queue_alloc
            .queue()
            .from_host(&cast(scheme.idx_strides()))
            .as_ptr();
        let c_strides = queue_alloc
            .queue()
            .from_host(&cast(scheme.c_strides()))
            .as_ptr();
        let a_strides = queue_alloc
            .queue()
            .from_host(&cast(scheme.a_strides()))
            .as_ptr();
        let b_strides = queue_alloc
            .queue()
            .from_host(&cast(scheme.b_strides()))
            .as_ptr();
        let params = cuda::params![
            args.c_base,
            c_strides,
            args.a_base,
            a_strides,
            args.b_base,
            b_strides,
            0,
            idx_strides,
            strids_size
        ];
        let block = gcd(self.max_threads_block, scheme.count());
        self.module.launch(
            CString::new(NAME).unwrap(),
            ((scheme.count() + block - 1) / block) as c_uint,
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
            c_layout: layout.clone(),
            c_base: null_mut(),
            a_layout: layout.clone(),
            a_base: null(),
            b_layout: layout.clone(),
            b_base: null(),
        }
    }
    fn args<H: Hardware>(
        dt: DigitLayout,
        n: usize,
        h: usize,
        w: usize,
        c_base: *mut H::Byte,
        a_base: *const H::Byte,
        b_base: *const H::Byte,
    ) -> Args<H> {
        Args {
            c_layout: TensorLayout::new_contiguous(dt, &[n, h, w]),
            c_base,
            a_layout: TensorLayout::new_contiguous(dt, &[n, h, w]),
            a_base,
            b_layout: TensorLayout::new_contiguous(dt, &[n, h, w]),
            b_base,
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
