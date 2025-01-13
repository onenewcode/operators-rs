use std::{ffi::CString, sync::Arc};

use super::{Args, LayerNorm};
use crate::{cuda::{Gpu, Handle, ModuleBox}, utils::gcd, ByteOf, LaunchError, QueueAlloc, SchemeError};

pub struct Operator {
    _handle: Arc<Handle>,
    max_threads_block: usize,
    module: Arc<ModuleBox>,
}
const NAME: &str = "gelu_f16";
const CODE: &str = include_str!("layer_norm.cuh");
impl LayerNorm<Gpu> for Operator {}

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
        args.meta()?;
        // let Meta { dt_w, dt_a, n, d } = args.meta()?;
        // let Args {
        //     y_layout,
        //     y_base,
        //     x_layout,
        //     x_base,
        //     scale_layout,
        //     scale_base,
        //     bias_layout,
        //     bias_base,
        //     epsilon,
        // } = args;
        // let &[nsy, dsy] = y_layout.strides() else {
        //     unreachable!()
        // };
        // let &[nsx, dsx] = x_layout.strides() else {
        //     unreachable!()
        // };
        // let &[dss] = scale_layout.strides() else {
        //     unreachable!()
        // };
        // let &[dsb] = bias_layout.strides() else {
        //     unreachable!()
        // };

        // get_static! {
        //     n   d
        //     nsy dsy
        //     nsx dsx
        //         dss
        //         dsb
        // }


        let params = cuda::params![];
        let block = gcd(self.max_threads_block, 1);

        self.module.launch(
            CString::new(NAME).unwrap(),
            block as u32,
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
    half *__restrict__ data){{
    gelu(data);
    }}"#
    )
}

#[cfg(test)]
mod test {
    use core::f32;
    use std::ptr::null;

    use super::{Args, Gpu, Operator};
    use crate::{dyn_, Hardware, Operator as _, TensorLayout};
    use digit_layout::{
        types::{F16, F64},
        DigitLayout,
    };

    fn dyn_args<H: Hardware>(dt: DigitLayout) -> Args<H> {
        use std::ptr::null_mut;
        let yx_layout = TensorLayout::new_dyn(dt, &[dyn_(); 2], &[dyn_(); 2]);
        let sb_layout = TensorLayout::new_dyn(dt, &[dyn_(); 1], &[dyn_(); 1]);
        Args {
            y_layout: yx_layout.clone(),
            y_base: null_mut(),
            x_layout: yx_layout.clone(),
            x_base: null(),
            scale_layout: sb_layout.clone(),
            scale_base: null(),
            bias_layout: sb_layout.clone(),
            bias_base: null(),
            epsilon: 0.1f32,
        }
    }
    fn args<H: Hardware>(dt: DigitLayout, n: usize, d: usize, y_base: *mut H::Byte,x_base:*const  H::Byte,scale_base:*const  H::Byte,bias_base:*const  H::Byte,epsilon: f32) -> Args<H> {
        let yx_layout = TensorLayout::new_contiguous(dt, &[n, d]);
        let sb_layout=TensorLayout::new_contiguous(dt, &[d]);
        Args {
            y_layout: yx_layout.clone(),
            y_base,
            x_layout: yx_layout.clone(),
            x_base,
            scale_layout: sb_layout.clone(),
            scale_base,
            bias_layout:  sb_layout.clone(),
            bias_base,
            epsilon,
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
            // cuda::cast_load,
            // test_utils::{Diff, ErrorCollector},
        };
        // use cuda::memcpy_d2h;
        // use half::f16;
        use rand::Rng;

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);

        let n = 1024;
        let d = 1024;
        let epsilon=1.0f32;
        cpu_op.scheme(&dyn_args(F64), 0).unwrap();
        gpu_op.scheme(&dyn_args(F16), 0).unwrap();
        let mut y = vec![0.0f64; n * d];
        let mut x = vec![0.0f64; n * d];
        let mut scale = vec![0.0f64; d];
        let mut bias = vec![0.0f64; d];
        rand::thread_rng().fill(&mut y[..]);
        rand::thread_rng().fill(&mut x[..]);
        rand::thread_rng().fill(&mut scale[..]);
        rand::thread_rng().fill(&mut bias[..]);
        // let data_ans = gpu.apply(|ctx| {
        //     let stream = ctx.stream();
        //     let mut data = cast_load(&data, f16::from_f64, &stream);
        //     gpu_op
        //         .launch(&args(F16, n, d, data.as_mut_ptr().cast()), &mut [], &stream)
        //         .unwrap();
        //     let mut host = vec![f16::ZERO; n * d];
        //     memcpy_d2h(&mut host, &data);
        //     host
        // });

        let mut data_ref = y;
        cpu_op
            .launch(
                &args(F64, n, d, data_ref.as_mut_ptr().cast(),x.as_mut_ptr().cast(),scale.as_ptr().cast(),bias.as_ptr().cast(),epsilon),
                &mut [],
                &ThisThread,
            )
            .unwrap();

        // let diff = data_ref
        //     .into_iter()
        //     .zip(data_ans)
        //     .map(|(a, b)| Diff::new(a, b.to_f64()))
        //     .collect::<Vec<_>>();

        // let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
        // diff.into_iter().for_each(|diff| ec.push(diff));
        // println!("{ec}");

        // let (out, count) = ec.summary();
        // assert!(out * 1000 <= count);
    }
}
