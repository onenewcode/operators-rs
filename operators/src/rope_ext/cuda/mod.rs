use super::{args::Meta, fill_pos, Args,  RopeExt, Seq, SinCosTable};
use crate::{
    cuda::{Gpu, Handle, ModuleBox},
    get_static, shape_not_support, strides_not_support, type_not_support, Blob, ByteOf,
    LaunchError, QueueAlloc, SchemeError,
};
use digit_layout::{types as ty, DigitLayout};
use std::{ffi::CString, sync::Arc};

pub struct Operator {

}



impl  RopeExt<Gpu> for Operator {
    fn build_sincos<QA>(
        _dt: DigitLayout,
        _nctx: usize,
        _dh: usize,
        queue_alloc: &QA,
    ) -> SinCosTable<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        SinCosTable {
            nctx: 0,
            mem: queue_alloc.alloc(0),
        }
    }

    fn build_pos<I, QA>(
        dt: digit_layout::DigitLayout,
        nt: usize,
        iter: I,
        queue_alloc: &QA,
    ) -> QA::DevMem
    where
        I: IntoIterator<Item = Seq>,
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
        let mut host = Blob::new(dt.nbytes() * nt);
        match dt {
            ty::U32 => fill_pos(host.as_mut_ptr().cast::<u32>(), nt, iter),
            ty::U64 => fill_pos(host.as_mut_ptr().cast::<u64>(), nt, iter),
            _ => todo!(),
        }

        let mut blob = queue_alloc.alloc(host.len());
        queue_alloc.queue().memcpy_h2d(&mut blob, &host);
        blob
    }
}

impl crate::Operator for Operator {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type Args = Args<Gpu>;

    fn new(node: &Self::TopoNode) -> Self {
        let cc = node.0.device().compute_capability();
        Self {
        
        }
    }

    fn scheme(
        &mut self,
        _args: &Self::Args,
        _max_workspace_size: usize,
    ) -> Result<usize, SchemeError> {
        Ok(0)
    }

    fn launch<QA>(
        &self,
        _args: &Self::Args,
        _workspace: &mut [ByteOf<Self::Hardware>],
        _queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Self::Hardware>,
    {
       todo!();
        Ok(())
    }
}
