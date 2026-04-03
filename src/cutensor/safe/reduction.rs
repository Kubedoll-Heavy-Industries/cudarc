use super::{result, sys};

/// Reduction operation descriptor.
pub struct ReductionDescriptor {
    pub(crate) handle: sys::cutensorHandle_t,
    pub(crate) desc: sys::cutensorOperationDescriptor_t,
}

impl Drop for ReductionDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { result::destroy_operation_descriptor(self.desc) }.ok();
        }
    }
}
