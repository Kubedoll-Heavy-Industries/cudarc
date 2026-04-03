use super::{result, sys};

/// Tensor descriptor wrapper.
pub struct TensorDescriptor {
    pub(crate) handle: sys::cutensorHandle_t,
    pub(crate) desc: sys::cutensorTensorDescriptor_t,
    pub(crate) num_modes: u32,
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { result::destroy_tensor_descriptor(self.desc) }.ok();
        }
    }
}

impl TensorDescriptor {
    pub fn num_modes(&self) -> u32 {
        self.num_modes
    }
}
