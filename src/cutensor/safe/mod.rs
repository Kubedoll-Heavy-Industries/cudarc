//! Safe Rust wrappers for cuTENSOR.
//!
//! # Usage
//!
//! ```ignore
//! use cudarc::cutensor::CuTensor;
//! use cudarc::driver::CudaContext;
//! use cudarc::driver::CudaSlice;
//!
//! let ctx = CudaContext::new(0)?;
//! let stream = ctx.default_stream();
//! let cutensor = CuTensor::new(stream.clone())?;
//!
//! // Create tensor descriptors
//! let extents = vec![32i64, 64];
//! let strides = vec![64, 1];
//! let desc = cutensor.create_tensor_descriptor(&extents, &strides, cudarc::cutensor::sys::cudaDataType_t::CUDA_R_32F)?;
//! ```
//!
//! # Key Concepts
//!
//! - **Handle**: Main cuTENSOR context, created per-stream
//! - **Tensor Descriptor**: Describes tensor layout (extent, stride, data type)
//! - **Operation Descriptor**: Describes the operation to perform
//! - **Plan**: Optimized execution plan, built from operation + preference
//! - **Workspace**: GPU memory for plan execution
//!
//! # Limitations
//!
//! This safe API provides basic tensor descriptor creation. Full contraction
//! and reduction operations require a compute descriptor which needs to be added
//! to the sys bindings. For now, use the result layer functions directly for
//! complex tensor operations.

mod contraction;
mod reduction;
mod tensor;

#[cfg(test)]
mod safe_test;

pub use contraction::*;
pub use reduction::*;
pub use tensor::*;

use std::sync::Arc;

use crate::driver::CudaStream;

pub use super::result::CutensorError;
use super::{result, sys};

/// Main cuTENSOR handle for tensor operations.
///
/// Create with [CuTensor::new] and use for all tensor operations.
/// The handle is associated with a specific stream.
#[derive(Debug)]
pub struct CuTensor {
    handle: sys::cutensorHandle_t,
    stream: Arc<CudaStream>,
}

unsafe impl Send for CuTensor {}
unsafe impl Sync for CuTensor {}

impl Drop for CuTensor {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.ok();
        }
    }
}

impl CuTensor {
    /// Create a new cuTENSOR handle associated with the given stream.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CutensorError> {
        let handle = result::create_handle()?;
        Ok(Self { handle, stream })
    }

    /// Returns the cuTENSOR library version as (major, minor, patch).
    pub fn version(&self) -> (usize, usize, usize) {
        result::get_version()
    }

    /// Returns the underlying handle.
    pub fn handle(&self) -> sys::cutensorHandle_t {
        self.handle
    }

    /// Returns the associated stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Creates a dense tensor descriptor.
    ///
    /// # Arguments
    /// * `extent` - Size of each mode
    /// * `stride` - Stride between elements in each mode (use nullptr for row-major)
    /// * `data_type` - CUDA data type (e.g., CUDA_R_32F, CUDA_R_16F)
    ///
    /// Pass `None` for stride to use row-major layout (stride[i] = extent[i+1] * ... * extent[n-1])
    pub fn create_tensor_descriptor(
        &self,
        extent: &[i64],
        stride: Option<&[i64]>,
        data_type: sys::cudaDataType_t,
    ) -> Result<TensorDescriptor, CutensorError> {
        let num_modes = extent.len() as u32;
        let alignment = 256; // Default 256-byte alignment

        // If stride is None, compute row-major strides
        let stride = match stride {
            Some(s) => s.to_vec(),
            None => {
                let mut s = vec![1i64; extent.len()];
                for i in (0..extent.len()).rev() {
                    if i == extent.len() - 1 {
                        s[i] = 1;
                    } else {
                        s[i] = s[i + 1] * extent[i + 1];
                    }
                }
                s
            }
        };

        let desc = unsafe {
            result::create_tensor_descriptor(
                self.handle,
                num_modes,
                extent.as_ptr(),
                stride.as_ptr(),
                data_type,
                alignment,
            )?
        };

        Ok(TensorDescriptor {
            handle: self.handle,
            desc,
            num_modes,
        })
    }
}
