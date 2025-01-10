use super::{args::Scheme, Add, Args};
use crate::{
    cuda::{Gpu, Handle, ModuleBox}, ByteOf, LaunchError, QueueAlloc, SchemeError
};
use std::{ffi::{c_uint, CString}, sync::Arc};

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
    {   let scheme = Scheme::new(args)?;
        let strids_size=scheme.idx_strides().len() as i32;
        let idx_strides = queue_alloc.queue().from_host(scheme.idx_strides()).as_ptr();
        let c_strides = queue_alloc.queue().from_host(scheme.c_strides()).as_ptr();
        let a_strides = queue_alloc.queue().from_host(scheme.a_strides()).as_ptr();
        let b_strides = queue_alloc.queue().from_host(scheme.b_strides()).as_ptr();
        let params = cuda::params![args.c_base,c_strides,args.a_base,a_strides,args.b_base,b_strides,0,idx_strides,strids_size];
        let block = self.max_threads_block;

        self.module.launch(
            CString::new(NAME).unwrap(),
            ((scheme.count()+block-1)/block) as c_uint,
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
        types::F16,
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
}

