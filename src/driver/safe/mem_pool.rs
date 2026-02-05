//! CUDA Memory Pool support for stream-ordered allocations.
//!
//! Memory pools provide efficient caching allocator behavior - allocations become
//! fast after warmup because freed memory is cached and reused.
//!
//! This module requires CUDA 11.2+ and devices that support memory pools.

use std::sync::Arc;

use crate::driver::{result, sys};

use super::{CudaContext, CudaSlice, CudaStream, DeviceRepr, DriverError, ValidAsZeroBits};
use std::marker::PhantomData;

/// A CUDA memory pool for stream-ordered allocations.
///
/// Memory pools provide caching allocator behavior - allocations are fast after warmup
/// because previously freed memory is cached and reused.
///
/// Create with [`CudaContext::create_mem_pool`] or get the default pool with
/// [`CudaContext::default_mem_pool`].
///
/// # Example
/// ```ignore
/// let ctx = CudaContext::new(0)?;
/// let pool = ctx.default_mem_pool()?;
///
/// // Set release threshold to keep 1GB reserved
/// pool.set_release_threshold(1024 * 1024 * 1024)?;
///
/// let stream = ctx.new_stream()?;
/// let data: CudaSlice<f32> = stream.alloc_from_pool(1000, &pool)?;
/// ```
pub struct CudaMemPool {
    pub(crate) cu_pool: sys::CUmemoryPool,
    pub(crate) ctx: Arc<CudaContext>,
    /// Whether this pool was created by us (and should be destroyed on drop)
    /// or is a default pool (which should not be destroyed).
    pub(crate) owned: bool,
}

unsafe impl Send for CudaMemPool {}
unsafe impl Sync for CudaMemPool {}

impl Drop for CudaMemPool {
    fn drop(&mut self) {
        // Only destroy pools we created, not default pools
        if self.owned {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx
                .record_err(unsafe { result::mem_pool::destroy(self.cu_pool) });
        }
    }
}

/// Configuration for creating a new memory pool.
#[derive(Debug, Clone, Default)]
pub struct MemPoolConfig {
    /// Bytes to keep reserved even when not in use.
    /// If `None`, uses the CUDA default.
    pub release_threshold: Option<u64>,
}

impl CudaMemPool {
    /// Returns the underlying CUDA memory pool handle.
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/destroy the pool pointer**, as it is still
    /// owned by the [`CudaMemPool`].
    pub fn cu_pool(&self) -> sys::CUmemoryPool {
        self.cu_pool
    }

    /// Returns the context this pool belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Trim the pool, releasing memory back to the OS.
    ///
    /// This releases unused memory until the pool contains at most `min_bytes_to_keep` bytes.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g6a9a5f1b4f0b7e5c0e6e0a7c9a6a8a8a)
    pub fn trim(&self, min_bytes_to_keep: usize) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::mem_pool::trim(self.cu_pool, min_bytes_to_keep) }
    }

    /// Get the current amount of reserved memory in bytes.
    ///
    /// Reserved memory is memory that the pool has obtained from the OS
    /// and is available for future allocations.
    pub fn reserved_mem(&self) -> Result<u64, DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe {
            result::mem_pool::get_attribute(
                self.cu_pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            )
        }
    }

    /// Get the current amount of used memory in bytes.
    ///
    /// Used memory is memory that is currently allocated from the pool.
    pub fn used_mem(&self) -> Result<u64, DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe {
            result::mem_pool::get_attribute(
                self.cu_pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            )
        }
    }

    /// Get the release threshold in bytes.
    ///
    /// The pool will not release memory below this threshold when trimming.
    pub fn release_threshold(&self) -> Result<u64, DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe {
            result::mem_pool::get_attribute(
                self.cu_pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            )
        }
    }

    /// Set the release threshold in bytes.
    ///
    /// The pool will not release memory below this threshold when trimming.
    /// Setting a higher threshold can improve performance by keeping memory
    /// reserved for future allocations.
    ///
    /// Use `u64::MAX` to prevent the pool from ever releasing memory back to the OS.
    pub fn set_release_threshold(&self, bytes: u64) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe {
            result::mem_pool::set_attribute(
                self.cu_pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                bytes,
            )
        }
    }

    /// Check whether the pool reuses memory following event dependencies.
    pub fn reuse_follow_event_dependencies(&self) -> Result<bool, DriverError> {
        self.ctx.bind_to_thread()?;
        let val = unsafe {
            result::mem_pool::get_attribute(
                self.cu_pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
            )
        }?;
        Ok(val != 0)
    }

    /// Set whether the pool should reuse memory following event dependencies.
    pub fn set_reuse_follow_event_dependencies(&self, enable: bool) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe {
            result::mem_pool::set_attribute(
                self.cu_pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                if enable { 1 } else { 0 },
            )
        }
    }

    /// Check whether the pool allows opportunistic memory reuse.
    pub fn reuse_allow_opportunistic(&self) -> Result<bool, DriverError> {
        self.ctx.bind_to_thread()?;
        let val = unsafe {
            result::mem_pool::get_attribute(
                self.cu_pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            )
        }?;
        Ok(val != 0)
    }

    /// Set whether the pool allows opportunistic memory reuse.
    pub fn set_reuse_allow_opportunistic(&self, enable: bool) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe {
            result::mem_pool::set_attribute(
                self.cu_pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                if enable { 1 } else { 0 },
            )
        }
    }

    /// Check whether the pool allows internal dependencies for memory reuse.
    pub fn reuse_allow_internal_dependencies(&self) -> Result<bool, DriverError> {
        self.ctx.bind_to_thread()?;
        let val = unsafe {
            result::mem_pool::get_attribute(
                self.cu_pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            )
        }?;
        Ok(val != 0)
    }

    /// Set whether the pool allows internal dependencies for memory reuse.
    pub fn set_reuse_allow_internal_dependencies(&self, enable: bool) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe {
            result::mem_pool::set_attribute(
                self.cu_pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
                if enable { 1 } else { 0 },
            )
        }
    }
}

impl CudaContext {
    /// Create a new memory pool on this device.
    ///
    /// The pool is configured for the device associated with this context.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g4f2c67c59a7adfe6ba65cca1c7b6fd45)
    pub fn create_mem_pool(
        self: &Arc<Self>,
        config: MemPoolConfig,
    ) -> Result<Arc<CudaMemPool>, DriverError> {
        self.bind_to_thread()?;

        let cu_pool = unsafe { result::mem_pool::create(self.cu_device, self.ordinal as i32) }?;

        let pool = Arc::new(CudaMemPool {
            cu_pool,
            ctx: self.clone(),
            owned: true,
        });

        // Apply configuration
        if let Some(threshold) = config.release_threshold {
            pool.set_release_threshold(threshold)?;
        }

        Ok(pool)
    }

    /// Get the default memory pool for this device.
    ///
    /// The default pool is managed by CUDA and should not be destroyed.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g26aa5b41f58e5cb8f9e5e9b1a3e8e8e8)
    pub fn default_mem_pool(self: &Arc<Self>) -> Result<Arc<CudaMemPool>, DriverError> {
        self.bind_to_thread()?;

        let cu_pool = unsafe { result::mem_pool::get_default(self.cu_device) }?;

        Ok(Arc::new(CudaMemPool {
            cu_pool,
            ctx: self.clone(),
            owned: false, // Default pool should not be destroyed
        }))
    }

    /// Get the current memory pool for this device.
    ///
    /// Returns the pool currently set as the device's default for new allocations.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8d4f6a4b6b9c5d7a0a0a0a0a0a0a0a0a)
    pub fn current_mem_pool(self: &Arc<Self>) -> Result<Arc<CudaMemPool>, DriverError> {
        self.bind_to_thread()?;

        let cu_pool = unsafe { result::mem_pool::get_current(self.cu_device) }?;

        Ok(Arc::new(CudaMemPool {
            cu_pool,
            ctx: self.clone(),
            owned: false, // We don't own pools retrieved this way
        }))
    }

    /// Set the current memory pool for this device.
    ///
    /// New allocations via `malloc_async` will use this pool.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g0c0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a)
    pub fn set_mem_pool(self: &Arc<Self>, pool: &CudaMemPool) -> Result<(), DriverError> {
        if pool.ctx.cu_device != self.cu_device {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_VALUE));
        }
        self.bind_to_thread()?;
        unsafe { result::mem_pool::set_current(self.cu_device, pool.cu_pool) }
    }
}

impl CudaStream {
    /// Allocate memory from a specific memory pool.
    ///
    /// The allocation is stream-ordered, meaning it will be available once all
    /// previous work on this stream completes.
    ///
    /// # Safety
    /// This is unsafe because the memory is unset.
    ///
    /// # Example
    /// ```ignore
    /// let pool = ctx.default_mem_pool()?;
    /// let data: CudaSlice<f32> = unsafe { stream.alloc_from_pool(1000, &pool)? };
    /// ```
    pub unsafe fn alloc_from_pool<T: ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
        pool: &CudaMemPool,
    ) -> Result<CudaSlice<T>, DriverError> {
        self.ctx.bind_to_thread()?;

        let num_bytes = len * std::mem::size_of::<T>();
        let cu_device_ptr = result::mem_pool::alloc_async(pool.cu_pool, num_bytes, self.cu_stream)?;

        let (read, write) = if self.ctx.is_event_tracking() {
            (
                Some(self.ctx.new_event(None)?),
                Some(self.ctx.new_event(None)?),
            )
        } else {
            (None, None)
        };

        Ok(CudaSlice {
            cu_device_ptr,
            len,
            read,
            write,
            stream: self.clone(),
            marker: PhantomData,
        })
    }

    /// Allocate zeroed memory from a specific memory pool.
    ///
    /// The allocation is stream-ordered and will be zeroed.
    ///
    /// # Example
    /// ```ignore
    /// let pool = ctx.default_mem_pool()?;
    /// let data: CudaSlice<f32> = stream.alloc_zeros_from_pool(1000, &pool)?;
    /// ```
    pub fn alloc_zeros_from_pool<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
        pool: &CudaMemPool,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc_from_pool(len, pool) }?;
        self.memset_zeros(&mut dst)?;
        Ok(dst)
    }
}
