use std::{ffi::CString, sync::Arc, usize};

use super::{Args, LayerNorm};
use crate::{
    cuda::{Gpu, Handle},
    get_static,
    layer_norm::args::Meta,
    utils::gcd,
    ByteOf, LaunchError, QueueAlloc, SchemeError,
};

pub struct Operator {
    _handle: Arc<Handle>,
    max_threads_block: usize,
}
const NAME: &str = "layer_norm_f32";
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
        let Meta { dt_w, dt_a, n, d } = args.meta()?;
        let Args {
            y_layout,
            y_base,
            x_layout,
            x_base,
            scale_layout,
            scale_base,
            bias_layout,
            bias_base,
            epsilon,
        } = args;
        let &[nsy, dsy] = y_layout.strides() else {
            unreachable!()
        };
        let &[nsx, dsx] = x_layout.strides() else {
            unreachable!()
        };
        let &[dss] = scale_layout.strides() else {
            unreachable!()
        };
        let &[dsb] = bias_layout.strides() else {
            unreachable!()
        };

        get_static! {
            n   d
            nsy dsy
            nsx dsx
                dss
                dsb
        }

        fn cast(strides: &[isize], size: usize) -> Vec<isize> {
            strides.iter().map(|x| x / size as isize).collect()
        }
        let &[nsy, dsy, nsx, dsx] = cast(&[nsy, dsy, nsx, dsx], dt_a.nbytes()).as_slice() else {
            unreachable!()
        };
        let &[dss, dsb] = cast(&[dss, dsb], dt_w.nbytes()).as_slice() else {
            unreachable!()
        };
        let params = cuda::params![
            y_base, x_base, scale_base, bias_base, epsilon, d, nsy, dsy, nsx, dsx, dss, dsb
        ];
        let block = gcd(self.max_threads_block, d);
        let dimx = (d + block - 1) / block;
        self._handle
            .compile_kernel(NAME, self._handle.device().compute_capability(), || {
                format_code(block, dimx)
            })
            .launch(
                CString::new(NAME).unwrap(),
                (n as _, dimx as _),
                block as u32,
                params.as_ptr(),
                0,
                queue_alloc.queue(),
            );
        Ok(())
    }
}

fn format_code(thread_dimx: usize, block_dimx: usize) -> String {
    format!(
        r#"{CODE}
    extern "C" __global__ void {NAME} (
    float *__restrict__ y,
    float const *__restrict__ x,
    float const *__restrict__ scale,
    float const *__restrict__ bias,
    float epsilon,
    int const d,
    int const nsy,
    int const dsy,
    int const nsx,
    int const dsx,
    int const dss,
    int const dsb)
{{
    layer_norm<{thread_dimx}, {block_dimx}>(y, x, scale, bias, epsilon, d, nsy, dsy, nsx, dsx, dss, dsb);
    }}"#
    )
}

#[cfg(test)]
mod test {
    use core::f32;
    use std::ptr::null;

    use super::{Args, Gpu, Operator};
    use crate::{dyn_, Hardware, Operator as _, TensorLayout};
    use digit_layout::{types::F32, DigitLayout};

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
    fn args<H: Hardware>(
        dt: DigitLayout,
        n: usize,
        d: usize,
        y_base: *mut H::Byte,
        x_base: *const H::Byte,
        scale_base: *const H::Byte,
        bias_base: *const H::Byte,
        epsilon: f32,
    ) -> Args<H> {
        let yx_layout = TensorLayout::new_contiguous(dt, &[n, d]);
        let sb_layout = TensorLayout::new_contiguous(dt, &[d]);
        Args {
            y_layout: yx_layout.clone(),
            y_base,
            x_layout: yx_layout.clone(),
            x_base,
            scale_layout: sb_layout.clone(),
            scale_base,
            bias_layout: sb_layout.clone(),
            bias_base,
            epsilon,
        }
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
        // use rand::Rng;

        let Some(gpu) = Gpu::init() else {
            return;
        };

        let mut cpu_op = RefOp::new(&Cpu);
        let mut gpu_op = Operator::new(&gpu);

        let n = 2;
        let d = 512;
        let epsilon = 1.0f32;
        cpu_op.scheme(&dyn_args(F32), 0).unwrap();
        gpu_op.scheme(&dyn_args(F32), 0).unwrap();
        let y = vec![0.0f32; n * d];
        let x = vec![1.0f32; n * d];
        let scale = vec![1.0f32; d];
        let bias = vec![1.0f32; d];
        // rand::thread_rng().fill(&mut y[..]);
        // rand::thread_rng().fill(&mut x[..]);
        // rand::thread_rng().fill(&mut scale[..]);
        // rand::thread_rng().fill(&mut bias[..]);
        let data_ans = gpu.apply(|ctx| {
            let stream = ctx.stream();
            let mut data = cast_load(&y, f32::from, &stream);
            let x = cast_load(&x, f32::from, &stream);
            let scale = cast_load(&scale, f32::from, &stream);
            let bias = cast_load(&bias, f32::from, &stream);
            gpu_op
                .launch(
                    &args(
                        F32,
                        n,
                        d,
                        data.as_mut_ptr().cast(),
                        x.as_ptr().cast(),
                        scale.as_ptr().cast(),
                        bias.as_ptr().cast(),
                        epsilon,
                    ),
                    &mut [],
                    &stream,
                )
                .unwrap();
            let mut host = vec![0.0f32; n * d];
            memcpy_d2h(&mut host, &data);
            host
        });

        let mut data_ref = y;
        cpu_op
            .launch(
                &args(
                    F32,
                    n,
                    d,
                    data_ref.as_mut_ptr().cast(),
                    x.as_ptr().cast(),
                    scale.as_ptr().cast(),
                    bias.as_ptr().cast(),
                    epsilon,
                ),
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let diff = data_ref
            .into_iter()
            .zip(data_ans)
            .map(|(a, b)| Diff::new(a as _, b as _))
            .collect::<Vec<_>>();

        let mut ec = ErrorCollector::new(f16::EPSILON.to_f64(), 0.);
        diff.into_iter().for_each(|diff| ec.push(diff));
        println!("{ec}");

        let (out, count) = ec.summary();
        assert!(out * 1000 <= count);
    }
}
