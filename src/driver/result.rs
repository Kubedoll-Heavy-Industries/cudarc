//! A thin wrapper around [sys].
//!
//! While all the functions here will return [Result], they are
//! mostly all still unsafe because order of operations
//! really matters.
//!
//! This also only exposes the `*_async` version of functions
//! because mixing the two is confusing and even more unsafe.
//!
//! This module also groups functions into sub-modules
//! to make naming easier. For example `sys::cuStreamCreate()`
//! turns into [stream::create()], where [stream] is a module.

use super::sys::{self};
use core::ffi::{c_uchar, c_uint, c_void, CStr};
use std::mem::MaybeUninit;

/// Wrapper around [sys::CUresult]. See
/// nvidia's [CUresult docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9)
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DriverError(pub sys::CUresult);

impl sys::CUresult {
    #[inline]
    pub fn result(self) -> Result<(), DriverError> {
        match self {
            sys::CUresult::CUDA_SUCCESS => Ok(()),
            _ => Err(DriverError(self)),
        }
    }
}

impl DriverError {
    /// Gets the name for this error.
    ///
    /// See [cuGetErrorName() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR_1g2c4ac087113652bb3d1f95bf2513c468)
    pub fn error_name(&self) -> Result<&CStr, DriverError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            sys::cuGetErrorName(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }

    /// Gets the error string for this error.
    ///
    /// See [cuGetErrorString() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html#group__CUDA__ERROR_1g72758fcaf05b5c7fac5c25ead9445ada)
    pub fn error_string(&self) -> Result<&CStr, DriverError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            sys::cuGetErrorString(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }
}

impl std::fmt::Debug for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.error_string() {
            Ok(err_str) => f
                .debug_tuple("DriverError")
                .field(&self.0)
                .field(&err_str)
                .finish(),
            Err(_) => f
                .debug_tuple("DriverError")
                .field(&self.0)
                .field(&"<Failure when calling cuGetErrorString()>")
                .finish(),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DriverError {}

/// Initializes the CUDA driver API.
/// **MUST BE CALLED BEFORE ANYTHING ELSE**
///
/// See [cuInit() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3)
pub fn init() -> Result<(), DriverError> {
    unsafe { sys::cuInit(0).result() }
}

pub mod device {
    //! Device management functions (`cuDevice*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE)

    use super::{
        sys::{self},
        DriverError,
    };
    use std::{
        ffi::{c_int, CStr},
        mem::MaybeUninit,
        string::String,
    };

    /// Get a device for a specific ordinal.
    /// See [cuDeviceGet() docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb).
    pub fn get(ordinal: c_int) -> Result<sys::CUdevice, DriverError> {
        let mut dev = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceGet(dev.as_mut_ptr(), ordinal).result()?;
            Ok(dev.assume_init())
        }
    }

    /// Gets the number of available devices.
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74)
    pub fn get_count() -> Result<c_int, DriverError> {
        let mut count = MaybeUninit::uninit();
        unsafe {
            sys::cuDeviceGetCount(count.as_mut_ptr()).result()?;
            Ok(count.assume_init())
        }
    }

    /// Returns the total amount of memory in bytes on the device.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d)
    ///
    /// # Safety
    /// Must be a device returned from [get].
    pub unsafe fn total_mem(dev: sys::CUdevice) -> Result<usize, DriverError> {
        let mut bytes = MaybeUninit::uninit();
        sys::cuDeviceTotalMem_v2(bytes.as_mut_ptr(), dev).result()?;
        Ok(bytes.assume_init())
    }

    /// Get an attribute of a device.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8c6e2c7b5c7c8b7e6f7f4c2b7f6d9c5d)
    ///
    /// # Safety
    /// Must be a device returned from [get].
    pub unsafe fn get_attribute(
        dev: sys::CUdevice,
        attrib: sys::CUdevice_attribute,
    ) -> Result<i32, DriverError> {
        let mut value = MaybeUninit::uninit();
        sys::cuDeviceGetAttribute(value.as_mut_ptr(), attrib, dev).result()?;
        Ok(value.assume_init())
    }

    /// Get name of the device.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f)
    pub fn get_name(dev: sys::CUdevice) -> Result<String, DriverError> {
        const BUF_SIZE: usize = 128;
        let mut buf = [0u8; BUF_SIZE];
        unsafe {
            sys::cuDeviceGetName(buf.as_mut_ptr() as _, BUF_SIZE as _, dev).result()?;
        }
        let name = CStr::from_bytes_until_nul(&buf).expect("No null byte was present");
        Ok(String::from_utf8_lossy(name.to_bytes()).into())
    }

    pub fn get_uuid(dev: sys::CUdevice) -> Result<sys::CUuuid, DriverError> {
        let id: sys::CUuuid;
        unsafe {
            let mut uuid = MaybeUninit::uninit();
            #[cfg(not(any(feature = "cuda-13000", feature = "cuda-13010")))]
            sys::cuDeviceGetUuid(uuid.as_mut_ptr(), dev).result()?;
            #[cfg(any(feature = "cuda-13000", feature = "cuda-13010"))]
            sys::cuDeviceGetUuid_v2(uuid.as_mut_ptr(), dev).result()?;
            id = uuid.assume_init();
        }
        Ok(id)
    }
}

pub mod function {
    use super::sys::{self, CUfunc_cache_enum, CUfunction_attribute_enum};
    use std::mem::MaybeUninit;

    /// Gets a specific attribute of a CUDA function.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn get_function_attribute(
        f: sys::CUfunction,
        attribute: CUfunction_attribute_enum,
    ) -> Result<i32, super::DriverError> {
        let mut value = MaybeUninit::uninit();
        unsafe {
            sys::cuFuncGetAttribute(value.as_mut_ptr(), attribute, f).result()?;
            Ok(value.assume_init())
        }
    }

    /// Sets the specific attribute of a cuda function.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g317e77d2657abf915fd9ed03e75f3eb0)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn set_function_attribute(
        f: sys::CUfunction,
        attribute: CUfunction_attribute_enum,
        value: i32,
    ) -> Result<(), super::DriverError> {
        unsafe {
            sys::cuFuncSetAttribute(f, attribute, value).result()?;
        }

        Ok(())
    }

    /// Sets the cache config of a CUDA function.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g6699ca1943ac2655effa0d571b2f4f15)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn set_function_cache_config(
        f: sys::CUfunction,
        attribute: CUfunc_cache_enum,
    ) -> Result<(), super::DriverError> {
        unsafe {
            sys::cuFuncSetCacheConfig(f, attribute).result()?;
        }

        Ok(())
    }
}

pub mod occupancy {

    use core::{
        ffi::{c_int, c_uint},
        mem::MaybeUninit,
    };

    use super::{
        sys::{self},
        DriverError,
    };

    /// Returns dynamic shared memory available per block when launching numBlocks blocks on SM.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gae02af6a9df9e1bbd51941af631bce69)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn available_dynamic_shared_mem_per_block(
        f: sys::CUfunction,
        num_blocks: c_int,
        block_size: c_int,
    ) -> Result<usize, DriverError> {
        let mut dynamic_smem_size = MaybeUninit::uninit();
        unsafe {
            sys::cuOccupancyAvailableDynamicSMemPerBlock(
                dynamic_smem_size.as_mut_ptr(),
                f,
                num_blocks,
                block_size,
            )
            .result()?;
        }
        Ok(dynamic_smem_size.assume_init())
    }

    /// Returns occupancy of a function.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98)
    ///
    /// # Safety
    /// Function must exist.
    pub unsafe fn max_active_block_per_multiprocessor(
        f: sys::CUfunction,
        block_size: c_int,
        dynamic_smem_size: usize,
    ) -> Result<i32, DriverError> {
        let mut num_blocks = MaybeUninit::uninit();
        unsafe {
            sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                num_blocks.as_mut_ptr(),
                f,
                block_size,
                dynamic_smem_size,
            )
            .result()?;
        }
        Ok(num_blocks.assume_init())
    }

    /// Returns occupancy of a function.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1g8f1da4d4983e5c3025447665423ae2c2)
    ///
    /// # Safety
    /// Function must exist. No invalid flags.
    pub unsafe fn max_active_block_per_multiprocessor_with_flags(
        f: sys::CUfunction,
        block_size: c_int,
        dynamic_smem_size: usize,
        flags: c_uint,
    ) -> Result<i32, DriverError> {
        let mut num_blocks = MaybeUninit::uninit();
        unsafe {
            sys::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                num_blocks.as_mut_ptr(),
                f,
                block_size,
                dynamic_smem_size,
                flags,
            )
            .result()?;
        }
        Ok(num_blocks.assume_init())
    }

    /// Suggest a launch configuration with reasonable occupancy.
    ///
    /// Returns (min_grid_size, block_size)
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gf179c4ab78962a8468e41c3f57851f03)
    ///
    /// # Safety
    /// Function must exist and the shared memory function must be correct.  No invalid flags.
    pub unsafe fn max_potential_block_size(
        f: sys::CUfunction,
        block_size_to_dynamic_smem_size: sys::CUoccupancyB2DSize,
        dynamic_smem_size: usize,
        block_size_limit: c_int,
    ) -> Result<(i32, i32), DriverError> {
        let mut min_grid_size = MaybeUninit::uninit();
        let mut block_size = MaybeUninit::uninit();
        unsafe {
            sys::cuOccupancyMaxPotentialBlockSize(
                min_grid_size.as_mut_ptr(),
                block_size.as_mut_ptr(),
                f,
                block_size_to_dynamic_smem_size,
                dynamic_smem_size,
                block_size_limit,
            )
            .result()?;
        }
        Ok((min_grid_size.assume_init(), block_size.assume_init()))
    }

    /// Suggest a launch configuration with reasonable occupancy.
    ///
    /// Returns (min_grid_size, block_size)
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1g04c0bb65630f82d9b99a5ca0203ee5aa)
    ///
    /// # Safety
    /// Function must exist and the shared memory function must be correct.  No invalid flags.
    pub unsafe fn max_potential_block_size_with_flags(
        f: sys::CUfunction,
        block_size_to_dynamic_smem_size: sys::CUoccupancyB2DSize,
        dynamic_smem_size: usize,
        block_size_limit: c_int,
        flags: c_uint,
    ) -> Result<(i32, i32), DriverError> {
        let mut min_grid_size = MaybeUninit::uninit();
        let mut block_size = MaybeUninit::uninit();
        unsafe {
            sys::cuOccupancyMaxPotentialBlockSizeWithFlags(
                min_grid_size.as_mut_ptr(),
                block_size.as_mut_ptr(),
                f,
                block_size_to_dynamic_smem_size,
                dynamic_smem_size,
                block_size_limit,
                flags,
            )
            .result()?;
        }
        Ok((min_grid_size.assume_init(), block_size.assume_init()))
    }
}

pub mod primary_ctx {
    //! Primary context management functions (`cuDevicePrimaryCtx*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX)

    use super::{
        sys::{self},
        DriverError,
    };
    use std::mem::MaybeUninit;

    /// Creates a primary context on the device and pushes it onto the primary context stack.
    /// Call [release] to free it.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300)
    ///
    /// # Safety
    ///
    /// This is only safe with a device that was returned from [super::device::get].
    pub unsafe fn retain(dev: sys::CUdevice) -> Result<sys::CUcontext, DriverError> {
        let mut ctx = MaybeUninit::uninit();
        sys::cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev).result()?;
        Ok(ctx.assume_init())
    }

    /// Release a reference to the current primary context.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad).
    ///
    /// # Safety
    ///
    /// This is only safe with a device that was returned from [super::device::get].
    pub unsafe fn release(dev: sys::CUdevice) -> Result<(), DriverError> {
        sys::cuDevicePrimaryCtxRelease_v2(dev).result()
    }
}

pub mod ctx {
    //! Context management functions (`cuCtx*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX)

    use super::{
        sys::{self},
        DriverError,
    };
    use std::mem::MaybeUninit;

    /// Binds the specified CUDA context to the calling CPU thread.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7)
    ///
    /// # Safety
    ///
    /// This has weird behavior depending on the value of `ctx`. See cuda docs for more info.
    /// In general this should only be called with an already initialized context,
    /// and one that wasn't already freed.
    pub unsafe fn set_current(ctx: sys::CUcontext) -> Result<(), DriverError> {
        sys::cuCtxSetCurrent(ctx).result()
    }

    /// Returns the CUDA context bound to the calling CPU thread if there is one.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0)
    pub fn get_current() -> Result<Option<sys::CUcontext>, DriverError> {
        let mut ctx = MaybeUninit::uninit();
        unsafe {
            sys::cuCtxGetCurrent(ctx.as_mut_ptr()).result()?;
            let ctx: sys::CUcontext = ctx.assume_init();
            if ctx.is_null() {
                Ok(None)
            } else {
                Ok(Some(ctx))
            }
        }
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g66655c37602c8628eae3e40c82619f1e)
    #[cfg(not(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000"
    )))]
    pub fn set_flags(flags: sys::CUctx_flags) -> Result<(), DriverError> {
        unsafe { sys::cuCtxSetFlags(flags as u32).result() }
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616)
    pub fn synchronize() -> Result<(), DriverError> {
        unsafe { sys::cuCtxSynchronize() }.result()
    }

    /// Gets the value of a context limit.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da4f8d6b7f6f06)
    pub fn get_limit(limit: sys::CUlimit) -> Result<usize, DriverError> {
        let mut value = MaybeUninit::uninit();
        unsafe {
            sys::cuCtxGetLimit(value.as_mut_ptr(), limit).result()?;
            Ok(value.assume_init())
        }
    }

    /// Sets the value of a context limit.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gf9496524a98e2ee1896d4b97d4c7ef32)
    pub fn set_limit(limit: sys::CUlimit, value: usize) -> Result<(), DriverError> {
        unsafe { sys::cuCtxSetLimit(limit, value).result() }
    }

    /// Gets the cache configuration preference.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76b6bc8f4d3b1f0d85e7)
    pub fn get_cache_config() -> Result<sys::CUfunc_cache, DriverError> {
        let mut config = MaybeUninit::uninit();
        unsafe {
            sys::cuCtxGetCacheConfig(config.as_mut_ptr()).result()?;
            Ok(config.assume_init())
        }
    }

    /// Sets the cache configuration preference.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g54699acb1e6b97eee1535e59a738229a)
    pub fn set_cache_config(config: sys::CUfunc_cache) -> Result<(), DriverError> {
        unsafe { sys::cuCtxSetCacheConfig(config).result() }
    }
}

pub mod stream {
    //! Stream management functions (`cuStream*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM).

    use super::{
        sys::{self},
        DriverError,
    };
    use std::mem::MaybeUninit;

    /// The kind of stream to initialize.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4)
    pub enum StreamKind {
        /// From cuda docs:
        /// > Default stream creation flag.
        Default,

        /// From cuda docs:
        /// > Specifies that work running in the created stream
        /// > may run concurrently with work in stream 0 (the NULL stream),
        /// > and that the created stream should perform no implicit
        /// > synchronization with stream 0.
        NonBlocking,
    }

    impl StreamKind {
        fn flags(self) -> sys::CUstream_flags {
            match self {
                Self::Default => sys::CUstream_flags::CU_STREAM_DEFAULT,
                Self::NonBlocking => sys::CUstream_flags::CU_STREAM_NON_BLOCKING,
            }
        }
    }

    /// The null stream, which is just a null pointer. **Recommend not using this.**
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream)
    pub fn null() -> sys::CUstream {
        std::ptr::null_mut()
    }

    /// Creates a stream with the specified kind.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4)
    pub fn create(kind: StreamKind) -> Result<sys::CUstream, DriverError> {
        let mut stream = MaybeUninit::uninit();
        unsafe {
            sys::cuStreamCreate(stream.as_mut_ptr(), kind.flags() as u32).result()?;
            Ok(stream.assume_init())
        }
    }

    /// Wait until a stream's tasks are completed.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad)
    ///
    /// # Safety
    ///
    /// This should only be called with stream created by [create] and not already
    /// destroyed. This follows default stream semantics, see relevant cuda docs.
    pub unsafe fn synchronize(stream: sys::CUstream) -> Result<(), DriverError> {
        sys::cuStreamSynchronize(stream).result()
    }

    /// Destroys a stream.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758)
    ///
    /// # Safety
    ///
    /// This should only be called with stream created by [create] and not already
    /// destroyed. This follows default stream semantics, see relevant cuda docs.
    pub unsafe fn destroy(stream: sys::CUstream) -> Result<(), DriverError> {
        sys::cuStreamDestroy_v2(stream).result()
    }

    /// Make a compute stream wait on an event.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f)
    ///
    /// # Safety
    /// 1. Both stream and event must not have been freed already
    pub unsafe fn wait_event(
        stream: sys::CUstream,
        event: sys::CUevent,
        flags: sys::CUevent_wait_flags,
    ) -> Result<(), DriverError> {
        sys::cuStreamWaitEvent(stream, event, flags as u32).result()
    }

    /// Attach managed memory to a stream.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533)
    ///
    /// # Safety
    /// See the cuda docs, there are a lot of considerations for this one.
    /// > Accessing memory on the device from streams that are not associated with it will produce undefined results. No error checking is performed by the Unified Memory system to ensure that kernels launched into other streams do not access this region.
    pub unsafe fn attach_mem_async(
        stream: sys::CUstream,
        dptr: sys::CUdeviceptr,
        num_bytes: usize,
        flags: sys::CUmemAttach_flags,
    ) -> Result<(), DriverError> {
        sys::cuStreamAttachMemAsync(stream, dptr, num_bytes, flags as u32).result()
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f)
    ///
    /// # Safety
    /// See docs, it's really unsafe ya'll.
    pub unsafe fn launch_host_function(
        stream: sys::CUstream,
        func: unsafe extern "C" fn(*mut ::core::ffi::c_void),
        arg: *mut std::ffi::c_void,
    ) -> Result<(), DriverError> {
        sys::cuLaunchHostFunc(stream, Some(func), arg).result()
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143)
    /// # Safety
    /// Stream must be valid
    pub unsafe fn begin_capture(
        stream: sys::CUstream,
        mode: sys::CUstreamCaptureMode,
    ) -> Result<(), DriverError> {
        sys::cuStreamBeginCapture_v2(stream, mode).result()
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c)
    /// # Safety
    /// Stream must be valid
    pub unsafe fn end_capture(stream: sys::CUstream) -> Result<sys::CUgraph, DriverError> {
        let mut graph = MaybeUninit::uninit();
        sys::cuStreamEndCapture(stream, graph.as_mut_ptr()).result()?;
        Ok(graph.assume_init())
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca)
    /// # Safety
    /// Stream must be valid
    pub unsafe fn is_capturing(
        stream: sys::CUstream,
    ) -> Result<sys::CUstreamCaptureStatus, DriverError> {
        let mut status = MaybeUninit::uninit();
        sys::cuStreamIsCapturing(stream, status.as_mut_ptr()).result()?;
        Ok(status.assume_init())
    }

    /// Information about ongoing stream capture
    #[derive(Debug, Clone)]
    pub struct CaptureInfo {
        pub status: sys::CUstreamCaptureStatus,
        pub id: u64,
    }

    /// Get capture info for a stream.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g9d22e54a0755b3b0e01dca4c9a9e70c8)
    ///
    /// # Safety
    /// Stream must be valid
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn get_capture_info(stream: sys::CUstream) -> Result<CaptureInfo, DriverError> {
        let mut status = MaybeUninit::uninit();
        let mut id = MaybeUninit::uninit();
        sys::cuStreamGetCaptureInfo_v2(
            stream,
            status.as_mut_ptr(),
            id.as_mut_ptr(),
            std::ptr::null_mut(), // graph - not needed
            std::ptr::null_mut(), // dependencies
            std::ptr::null_mut(), // numDependencies
        )
        .result()?;
        Ok(CaptureInfo {
            status: status.assume_init(),
            id: id.assume_init(),
        })
    }
}

/// Allocates memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f)
///
/// # Safety
/// 1. The stream should be an already created stream.
/// 2. The memory return by this is unset, which may be invalid for `T`.
/// 3. All uses of this memory must be on the same stream.
pub unsafe fn malloc_async(
    stream: sys::CUstream,
    num_bytes: usize,
) -> Result<sys::CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    sys::cuMemAllocAsync(dev_ptr.as_mut_ptr(), num_bytes, stream).result()?;
    Ok(dev_ptr.assume_init())
}

/// Allocates memory
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467)
///
/// # Safety
/// 1. The memory return by this is unset, which may be invalid for `T`.
pub unsafe fn malloc_sync(num_bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    sys::cuMemAlloc_v2(dev_ptr.as_mut_ptr(), num_bytes).result()?;
    Ok(dev_ptr.assume_init())
}

/// Allocates managed memory.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32)
///
/// # Safety
/// 1. The memory return by this is unset, which may be invalid for `T`.
pub unsafe fn malloc_managed(
    num_bytes: usize,
    flags: sys::CUmemAttach_flags,
) -> Result<sys::CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    sys::cuMemAllocManaged(dev_ptr.as_mut_ptr(), num_bytes, flags as u32).result()?;
    Ok(dev_ptr.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9)
/// # Safety
/// 1. The memory return by this is unset, which may be invalid for `T`.
pub unsafe fn malloc_host(num_bytes: usize, flags: c_uint) -> Result<*mut c_void, DriverError> {
    let mut host_ptr = MaybeUninit::uninit();
    sys::cuMemHostAlloc(host_ptr.as_mut_ptr(), num_bytes, flags).result()?;
    Ok(host_ptr.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c)
/// # Safety
/// 1. `host_ptr` must have been returned by [malloc_host]
/// 2. `host_ptr` should not be null.
pub unsafe fn free_host(host_ptr: *mut c_void) -> Result<(), DriverError> {
    sys::cuMemFreeHost(host_ptr).result()
}

/// Advise about the usage of a given memory range.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g27608c857a9254789c13f3e3b72029e2)
/// **Only available in 12.2+.
///
/// # Safety
/// 1. Memory must have been allocated by [malloc_managed()]
/// 2. num_bytes must be the amount of bytes passed to [malloc_managed()]
#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010"
)))]
pub unsafe fn mem_advise(
    dptr: sys::CUdeviceptr,
    num_bytes: usize,
    advice: sys::CUmem_advise,
    location: sys::CUmemLocation,
) -> Result<(), DriverError> {
    sys::cuMemAdvise_v2(dptr, num_bytes, advice, location).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gfe94f8b7fb56291ebcea44261aa4cb84)
/// **Only available in 12.2+.
///
/// # Safety
/// 1. The dptr/num_bytes must be allocated by [malloc_managed()] and must be the exact same memory range.
#[cfg(not(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010"
)))]
pub unsafe fn mem_prefetch_async(
    dptr: sys::CUdeviceptr,
    num_bytes: usize,
    location: sys::CUmemLocation,
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    sys::cuMemPrefetchAsync_v2(dptr, num_bytes, location, 0, stream).result()
}

/// Frees memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf)
///
/// # Safety
/// 1. The stream should be an already created stream.
/// 2. The memory should have been allocated on this stream.
/// 3. The memory should not have been freed already (double free)
pub unsafe fn free_async(dptr: sys::CUdeviceptr, stream: sys::CUstream) -> Result<(), DriverError> {
    sys::cuMemFreeAsync(dptr, stream).result()
}

/// Allocates memory
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a)
///
/// # Safety
/// 1. The memory should have been allocated with malloc_sync
pub unsafe fn free_sync(dptr: sys::CUdeviceptr) -> Result<(), DriverError> {
    sys::cuMemFree_v2(dptr).result()
}

/// Frees device memory.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a)
///
/// # Safety
/// 1. Memory must only be freed once.
/// 2. All async accesses to this pointer must have been completed.
pub unsafe fn memory_free(device_ptr: sys::CUdeviceptr) -> Result<(), DriverError> {
    sys::cuMemFree_v2(device_ptr).result()
}

/// Sets device memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627)
///
/// # Safety
/// 1. The resulting memory pattern may not be valid for `T`.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memset_d8_async(
    dptr: sys::CUdeviceptr,
    uc: c_uchar,
    num_bytes: usize,
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    sys::cuMemsetD8Async(dptr, uc, num_bytes, stream).result()
}

/// Sets device memory with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b)
///
/// # Safety
/// 1. The resulting memory pattern may not be valid for `T`.
/// 2. The device pointer should not have been freed already (double free)
pub unsafe fn memset_d8_sync(
    dptr: sys::CUdeviceptr,
    uc: c_uchar,
    num_bytes: usize,
) -> Result<(), DriverError> {
    sys::cuMemsetD8_v2(dptr, uc, num_bytes).result()
}

/// Copies memory from Host to Device with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169)
///
/// # Safety
/// **This function is asynchronous** in most cases, so the data from `src`
/// will be copied at a later point after this function returns.
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
/// 4. `src` must not be moved
pub unsafe fn memcpy_htod_async<T>(
    dst: sys::CUdeviceptr,
    src: &[T],
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    sys::cuMemcpyHtoDAsync_v2(
        dst,
        src.as_ptr() as *const _,
        std::mem::size_of_val(src),
        stream,
    )
    .result()
}

/// Copies memory from Host to Device
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169)
///
/// # Safety
/// **This function is synchronous**///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. `src` must not be moved
pub unsafe fn memcpy_htod_sync<T>(dst: sys::CUdeviceptr, src: &[T]) -> Result<(), DriverError> {
    sys::cuMemcpyHtoD_v2(dst, src.as_ptr() as *const _, std::mem::size_of_val(src)).result()
}

/// Copies memory from Device to Host with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362)
///
/// # Safety
/// **This function is asynchronous** in most cases, so `dst` will be
/// mutated at a later point after this function returns.
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memcpy_dtoh_async<T>(
    dst: &mut [T],
    src: sys::CUdeviceptr,
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    sys::cuMemcpyDtoHAsync_v2(
        dst.as_mut_ptr() as *mut _,
        src,
        std::mem::size_of_val(dst),
        stream,
    )
    .result()
}

/// Copies memory from Device to Host with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893)
///
/// # Safety
/// **This function is synchronous**
///
/// 1. `T` must be the type that device pointer was allocated with.
/// 2. The device pointer should not have been freed already (double free)
pub unsafe fn memcpy_dtoh_sync<T>(dst: &mut [T], src: sys::CUdeviceptr) -> Result<(), DriverError> {
    sys::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut _, src, std::mem::size_of_val(dst)).result()
}

/// Copies memory from Device to Device with stream ordered semantics.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8)
///
/// # Safety
/// 1. `T` must be the type that BOTH device pointers were allocated with.
/// 2. Neither device pointer should not have been freed already (double free)
/// 3. The stream should be the stream the memory was allocated on.
pub unsafe fn memcpy_dtod_async(
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    num_bytes: usize,
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    sys::cuMemcpyDtoDAsync_v2(dst, src, num_bytes, stream).result()
}

/// Copies memory from Device to Device
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b)
///
/// # Safety
/// 1. `T` must be the type that BOTH device pointers were allocated with.
/// 2. Neither device pointer should not have been freed already (double free)
pub unsafe fn memcpy_dtod_sync(
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    num_bytes: usize,
) -> Result<(), DriverError> {
    sys::cuMemcpyDtoD_v2(dst, src, num_bytes).result()
}

/// Copies device memory between two contexts asynchronously.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g82fcecb38018e64b98616a8ac30112f2)
///
/// # Safety
/// 1. Neither device pointer should have been freed (double free)
pub unsafe fn memcpy_peer_async(
    dst_ctx: sys::CUcontext,
    dst: sys::CUdeviceptr,
    src_ctx: sys::CUcontext,
    src: sys::CUdeviceptr,
    num_bytes: usize,
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    sys::cuMemcpyPeerAsync(dst, dst_ctx, src, src_ctx, num_bytes, stream).result()
}

/// Returns (free, total) memory in bytes.
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0)
pub fn mem_get_info() -> Result<(usize, usize), DriverError> {
    let mut free = 0;
    let mut total = 0;
    unsafe { sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _) }.result()?;
    Ok((free, total))
}

pub mod module {
    //! Module management functions (`cuModule*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE)

    use super::{
        sys::{self},
        DriverError,
    };
    use core::ffi::c_void;
    use std::ffi::CString;
    use std::mem::MaybeUninit;

    /// Loads a compute module from a given file.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3)
    pub fn load(fname: CString) -> Result<sys::CUmodule, DriverError> {
        let fname_ptr = fname.as_c_str().as_ptr();
        let mut module = MaybeUninit::uninit();
        unsafe {
            sys::cuModuleLoad(module.as_mut_ptr(), fname_ptr).result()?;
            Ok(module.assume_init())
        }
    }

    /// Load a module's data:
    ///
    /// > The pointer may be obtained by mapping a cubin or PTX or fatbin file,
    /// > passing a cubin or PTX or fatbin file as a NULL-terminated text string,
    /// > or incorporating a cubin or fatbin object into the executable resources
    /// > and using operating system calls such as Windows FindResource() to obtain the pointer.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b)
    ///
    /// # Safety
    /// The image must be properly formed pointer
    pub unsafe fn load_data(image: *const c_void) -> Result<sys::CUmodule, DriverError> {
        let mut module = MaybeUninit::uninit();
        sys::cuModuleLoadData(module.as_mut_ptr(), image).result()?;
        Ok(module.assume_init())
    }

    /// Returns a function handle from the given module.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf)
    ///
    /// # Safety
    /// `module` must be a properly allocated and not freed module.
    pub unsafe fn get_function(
        module: sys::CUmodule,
        name: CString,
    ) -> Result<sys::CUfunction, DriverError> {
        let name_ptr = name.as_c_str().as_ptr();
        let mut func = MaybeUninit::uninit();
        sys::cuModuleGetFunction(func.as_mut_ptr(), module, name_ptr).result()?;
        Ok(func.assume_init())
    }

    /// Returns a pointer to a global/constant symbol in the module.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43972c23c2d5c8a662f2d9a4d0c24)
    ///
    /// # Safety
    /// `module` must be a properly allocated and not freed module.
    pub unsafe fn get_global(
        module: sys::CUmodule,
        name: CString,
    ) -> Result<(sys::CUdeviceptr, usize), DriverError> {
        let name_ptr = name.as_c_str().as_ptr();
        let mut dptr = MaybeUninit::uninit();
        let mut bytes = MaybeUninit::uninit();
        sys::cuModuleGetGlobal_v2(dptr.as_mut_ptr(), bytes.as_mut_ptr(), module, name_ptr)
            .result()?;
        Ok((dptr.assume_init(), bytes.assume_init()))
    }

    /// Unloads a module.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b)
    ///
    /// # Safety
    /// `module` must not have be unloaded already.
    pub unsafe fn unload(module: sys::CUmodule) -> Result<(), DriverError> {
        sys::cuModuleUnload(module).result()
    }
}

pub mod event {
    use super::{
        sys::{self},
        DriverError,
    };
    use std::mem::MaybeUninit;

    /// Creates an event.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db)
    pub fn create(flags: sys::CUevent_flags) -> Result<sys::CUevent, DriverError> {
        let mut event = MaybeUninit::uninit();
        unsafe {
            sys::cuEventCreate(event.as_mut_ptr(), flags as u32).result()?;
            Ok(event.assume_init())
        }
    }

    /// Records an event.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1)
    ///
    /// # Safety
    /// This function is unsafe because event can be a null event, in which case
    pub unsafe fn record(event: sys::CUevent, stream: sys::CUstream) -> Result<(), DriverError> {
        unsafe { sys::cuEventRecord(event, stream).result() }
    }

    /// Computes the elapsed time (in milliseconds) between two events.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97)
    /// # Safety
    /// 1. Events must have been created by [create]
    /// 2. They should be on the same stream
    /// 3. They must not have been destroyed.
    pub unsafe fn elapsed(start: sys::CUevent, end: sys::CUevent) -> Result<f32, DriverError> {
        let mut ms: f32 = 0.0;
        unsafe {
            #[cfg(not(any(feature = "cuda-13000", feature = "cuda-13010")))]
            sys::cuEventElapsedTime((&mut ms) as *mut _, start, end).result()?;
            #[cfg(any(feature = "cuda-13000", feature = "cuda-13010"))]
            sys::cuEventElapsedTime_v2((&mut ms) as *mut _, start, end).result()?;
        }
        Ok(ms)
    }

    /// Queries an event's status.
    /// Returns `Ok` if all captured work has been completed, or `Err`: `CUDA_ERROR_NOT_READY` if
    /// any captured work is incomplete.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef)
    ///
    /// # Safety
    /// This function is unsafe because event can be a null event, in which case
    pub unsafe fn query(event: sys::CUevent) -> Result<(), DriverError> {
        unsafe { sys::cuEventQuery(event).result() }
    }

    /// Waits for an event to complete.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c)
    ///
    /// # Safety
    /// This function is unsafe because event can be a null event, in which case
    pub unsafe fn synchronize(event: sys::CUevent) -> Result<(), DriverError> {
        unsafe { sys::cuEventSynchronize(event).result() }
    }

    /// Destroys an event.
    ///
    /// > An event may be destroyed before it is complete (i.e., while cuEventQuery() would return CUDA_ERROR_NOT_READY).
    /// > In this case, the call does not block on completion of the event,
    /// > and any associated resources will automatically be released asynchronously at completion.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef)
    ///
    /// # Safety
    /// 1. Event must not have been freed already
    pub unsafe fn destroy(event: sys::CUevent) -> Result<(), DriverError> {
        sys::cuEventDestroy_v2(event).result()
    }
}

/// Launches a cuda functions
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15)
///
/// # Safety
/// This method is **very unsafe**.
///
/// 1. The cuda function must be a valid handle returned from a non-unloaded module.
/// 2. This is asynchronous, so the results of calling this function happen
///    at a later point after this function returns.
/// 3. All parameters used for this kernel should have been allocated by stream (I think?)
/// 4. The cuda kernel has mutable access to every parameter, that means every parameter
///    can change at a later point after callign this function. *Even non-mutable references*.
#[inline]
pub unsafe fn launch_kernel(
    f: sys::CUfunction,
    grid_dim: (c_uint, c_uint, c_uint),
    block_dim: (c_uint, c_uint, c_uint),
    shared_mem_bytes: c_uint,
    stream: sys::CUstream,
    kernel_params: &mut [*mut c_void],
) -> Result<(), DriverError> {
    sys::cuLaunchKernel(
        f,
        grid_dim.0,
        grid_dim.1,
        grid_dim.2,
        block_dim.0,
        block_dim.1,
        block_dim.2,
        shared_mem_bytes,
        stream,
        kernel_params.as_mut_ptr(),
        std::ptr::null_mut(),
    )
    .result()
}

/// Launches a cuda functions
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g06d753134145c4584c0c62525c1894cb)
///
/// # Safety
/// This method is **very unsafe**.
///
/// 1. The cuda function must be a valid handle returned from a non-unloaded module.
/// 2. This is asynchronous, so the results of calling this function happen
///    at a later point after this function returns.
/// 3. All parameters used for this kernel should have been allocated by stream (I think?)
/// 4. The cuda kernel has mutable access to every parameter, that means every parameter
///    can change at a later point after callign this function. *Even non-mutable references*.
#[inline]
pub unsafe fn launch_cooperative_kernel(
    f: sys::CUfunction,
    grid_dim: (c_uint, c_uint, c_uint),
    block_dim: (c_uint, c_uint, c_uint),
    shared_mem_bytes: c_uint,
    stream: sys::CUstream,
    kernel_params: &mut [*mut c_void],
) -> Result<(), DriverError> {
    sys::cuLaunchCooperativeKernel(
        f,
        grid_dim.0,
        grid_dim.1,
        grid_dim.2,
        block_dim.0,
        block_dim.1,
        block_dim.2,
        shared_mem_bytes,
        stream,
        kernel_params.as_mut_ptr(),
    )
    .result()
}

pub mod external_memory {
    use std::mem::MaybeUninit;

    use super::{
        sys::{self},
        DriverError,
    };

    /// Imports an external memory object, in this case an OpaqueFd.
    ///
    /// The memory should be destroyed using [`destroy_external_memory`].
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735)
    ///
    /// # Safety
    /// `size` must be the size of the size of the memory object in bytes.
    #[cfg(unix)]
    pub unsafe fn import_external_memory_opaque_fd(
        fd: std::os::fd::RawFd,
        size: u64,
    ) -> Result<sys::CUexternalMemory, DriverError> {
        let mut external_memory = MaybeUninit::uninit();
        let handle_description = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 { fd },
            size,
            flags: 0,
            reserved: [0; 16],
        };
        sys::cuImportExternalMemory(external_memory.as_mut_ptr(), &handle_description).result()?;
        Ok(external_memory.assume_init())
    }

    /// Imports an external memory object, in this case an OpaqueWin32 handle.
    ///
    /// The memory should be destroyed using [`destroy_external_memory`].
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735)
    ///
    /// # Safety
    /// `size` must be the size of the size of the memory object in bytes.
    #[cfg(windows)]
    pub unsafe fn import_external_memory_opaque_win32(
        handle: std::os::windows::io::RawHandle,
        size: u64,
    ) -> Result<sys::CUexternalMemory, DriverError> {
        let mut external_memory = MaybeUninit::uninit();
        let handle_description = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
            handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
                win32: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                    handle,
                    name: std::ptr::null(),
                },
            },
            size,
            flags: 0,
            reserved: [0; 16],
        };
        sys::cuImportExternalMemory(external_memory.as_mut_ptr(), &handle_description).result()?;
        Ok(external_memory.assume_init())
    }

    /// Destroys an external memory object.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g1b586dda86565617e7e0883b956c7052)
    ///
    /// # Safety
    /// 1. Any mapped buffers onto this object must already be freed.
    /// 2. The external memory must only be destroyed once.
    pub unsafe fn destroy_external_memory(
        external_memory: sys::CUexternalMemory,
    ) -> Result<(), DriverError> {
        sys::cuDestroyExternalMemory(external_memory).result()
    }

    /// Maps a buffer onto an imported memory object.
    ///
    /// The buffer must be freed using [`memory_free`](super::memory_free).
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1gb9fec33920400c70961b4e33d838da91)
    ///
    /// # Safety
    /// Mapped buffers may overlap.
    pub unsafe fn get_mapped_buffer(
        external_memory: sys::CUexternalMemory,
        offset: u64,
        size: u64,
    ) -> Result<sys::CUdeviceptr, DriverError> {
        let mut device_ptr = MaybeUninit::uninit();
        let buffer_description = sys::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset,
            size,
            flags: 0,
            reserved: [0; 16],
        };
        sys::cuExternalMemoryGetMappedBuffer(
            device_ptr.as_mut_ptr(),
            external_memory,
            &buffer_description,
        )
        .result()?;
        Ok(device_ptr.assume_init())
    }
}

pub mod graph {
    use super::*;

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1)
    /// # Safety
    /// graph must be valid
    pub unsafe fn instantiate(
        graph: sys::CUgraph,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<sys::CUgraphExec, DriverError> {
        let mut graph_exec = MaybeUninit::uninit();
        sys::cuGraphInstantiateWithFlags(graph_exec.as_mut_ptr(), graph, flags as u32 as u64)
            .result()?;
        Ok(graph_exec.assume_init())
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7)
    /// # Safety
    /// graph_exec must be valid
    pub unsafe fn exec_destroy(graph_exec: sys::CUgraphExec) -> Result<(), DriverError> {
        sys::cuGraphExecDestroy(graph_exec).result()
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g718cfd9681f078693d4be2426fd689c8)
    /// # Safety
    /// graph must be valid
    pub unsafe fn destroy(graph: sys::CUgraph) -> Result<(), DriverError> {
        sys::cuGraphDestroy(graph).result()
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471)
    /// # Safety
    /// graph & stream must be valid
    pub unsafe fn launch(
        graph_exec: sys::CUgraphExec,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuGraphLaunch(graph_exec, stream).result()
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gdb81438b083d42a26693f6f2bce150cd)
    /// # Safety
    /// graph_exec and stream must be valid
    pub unsafe fn upload(
        graph_exec: sys::CUgraphExec,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuGraphUpload(graph_exec, stream).result()
    }

    /// Returns all nodes in a graph.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g048f6e36f5d7e0ad5f6e2ab38ee37e55)
    ///
    /// # Safety
    /// graph must be valid
    pub unsafe fn get_nodes(graph: sys::CUgraph) -> Result<Vec<sys::CUgraphNode>, DriverError> {
        // First call to get the number of nodes
        let mut num_nodes = MaybeUninit::uninit();
        sys::cuGraphGetNodes(graph, std::ptr::null_mut(), num_nodes.as_mut_ptr()).result()?;
        let num_nodes = num_nodes.assume_init();

        if num_nodes == 0 {
            return Ok(Vec::new());
        }

        // Second call to get the actual nodes
        let mut nodes = vec![std::ptr::null_mut(); num_nodes];
        let mut actual_count = num_nodes;
        sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut actual_count).result()?;
        nodes.truncate(actual_count);
        Ok(nodes)
    }

    /// Returns all root nodes in a graph (nodes with no dependencies).
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00216ee8e72ca27c85c27e3e81e837f6)
    ///
    /// # Safety
    /// graph must be valid
    pub unsafe fn get_root_nodes(
        graph: sys::CUgraph,
    ) -> Result<Vec<sys::CUgraphNode>, DriverError> {
        // First call to get the number of root nodes
        let mut num_nodes = MaybeUninit::uninit();
        sys::cuGraphGetRootNodes(graph, std::ptr::null_mut(), num_nodes.as_mut_ptr()).result()?;
        let num_nodes = num_nodes.assume_init();

        if num_nodes == 0 {
            return Ok(Vec::new());
        }

        // Second call to get the actual nodes
        let mut nodes = vec![std::ptr::null_mut(); num_nodes];
        let mut actual_count = num_nodes;
        sys::cuGraphGetRootNodes(graph, nodes.as_mut_ptr(), &mut actual_count).result()?;
        nodes.truncate(actual_count);
        Ok(nodes)
    }

    /// Returns all edges in a graph as (from, to) pairs.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge9d27a6b2ebca4d9e5f94c0c8c8b0e06)
    ///
    /// # Safety
    /// graph must be valid
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn get_edges(
        graph: sys::CUgraph,
    ) -> Result<Vec<(sys::CUgraphNode, sys::CUgraphNode)>, DriverError> {
        // First call to get the number of edges
        let mut num_edges = MaybeUninit::uninit();
        sys::cuGraphGetEdges(
            graph,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            num_edges.as_mut_ptr(),
        )
        .result()?;
        let num_edges = num_edges.assume_init();

        if num_edges == 0 {
            return Ok(Vec::new());
        }

        // Second call to get the actual edges
        let mut from_nodes = vec![std::ptr::null_mut(); num_edges];
        let mut to_nodes = vec![std::ptr::null_mut(); num_edges];
        let mut actual_count = num_edges;
        sys::cuGraphGetEdges(
            graph,
            from_nodes.as_mut_ptr(),
            to_nodes.as_mut_ptr(),
            &mut actual_count,
        )
        .result()?;
        from_nodes.truncate(actual_count);
        to_nodes.truncate(actual_count);

        Ok(from_nodes.into_iter().zip(to_nodes).collect())
    }

    /// Clones a graph.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g9d5cfeb00b8ee918ea3c6f0816b4d8ef)
    ///
    /// # Safety
    /// graph must be valid
    pub unsafe fn clone(graph: sys::CUgraph) -> Result<sys::CUgraph, DriverError> {
        let mut cloned_graph = MaybeUninit::uninit();
        sys::cuGraphClone(cloned_graph.as_mut_ptr(), graph).result()?;
        Ok(cloned_graph.assume_init())
    }

    /// Returns the type of a graph node.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g65be75993be27f5c46ee30a3d62203c2)
    ///
    /// # Safety
    /// node must be valid
    pub unsafe fn node_get_type(
        node: sys::CUgraphNode,
    ) -> Result<sys::CUgraphNodeType, DriverError> {
        let mut node_type = MaybeUninit::uninit();
        sys::cuGraphNodeGetType(node, node_type.as_mut_ptr()).result()?;
        Ok(node_type.assume_init())
    }

    /// Sets the parameters of a kernel node in an instantiated graph.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c)
    ///
    /// # Safety
    /// graph_exec, node, and node_params must be valid.
    /// The kernel parameters (args) must match the kernel signature and remain valid.
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn exec_kernel_node_set_params(
        graph_exec: sys::CUgraphExec,
        node: sys::CUgraphNode,
        node_params: *const sys::CUDA_KERNEL_NODE_PARAMS,
    ) -> Result<(), DriverError> {
        sys::cuGraphExecKernelNodeSetParams(graph_exec, node, node_params).result()
    }

    /// Sets the parameters of a kernel node in an instantiated graph.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c)
    ///
    /// # Safety
    /// graph_exec, node, and node_params must be valid.
    /// The kernel parameters (args) must match the kernel signature and remain valid.
    #[cfg(any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    ))]
    pub unsafe fn exec_kernel_node_set_params(
        graph_exec: sys::CUgraphExec,
        node: sys::CUgraphNode,
        node_params: *const sys::CUDA_KERNEL_NODE_PARAMS,
    ) -> Result<(), DriverError> {
        sys::cuGraphExecKernelNodeSetParams_v2(graph_exec, node, node_params).result()
    }

    /// Sets the parameters of a memcpy node in an instantiated graph.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50a5c0a1a5a6b0c7b3e3d5a8a9c3b0d7)
    ///
    /// # Safety
    /// graph_exec, node, copy_params, and ctx must be valid.
    /// The source and destination memory must remain valid.
    pub unsafe fn exec_memcpy_node_set_params(
        graph_exec: sys::CUgraphExec,
        node: sys::CUgraphNode,
        copy_params: *const sys::CUDA_MEMCPY3D,
        ctx: sys::CUcontext,
    ) -> Result<(), DriverError> {
        sys::cuGraphExecMemcpyNodeSetParams(graph_exec, node, copy_params, ctx).result()
    }

    /// Updates an instantiated graph to match a new graph definition.
    ///
    /// Returns the update result and optionally an error node if the update failed.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g27a7df53a4a5e4a9c3d4d3b5a8a9c3b0)
    ///
    /// # Safety
    /// graph_exec and graph must be valid
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn exec_update(
        graph_exec: sys::CUgraphExec,
        graph: sys::CUgraph,
    ) -> Result<(sys::CUgraphExecUpdateResult, sys::CUgraphNode), DriverError> {
        let mut error_node = MaybeUninit::uninit();
        let mut update_result = MaybeUninit::uninit();
        sys::cuGraphExecUpdate(
            graph_exec,
            graph,
            error_node.as_mut_ptr(),
            update_result.as_mut_ptr(),
        )
        .result()?;
        Ok((update_result.assume_init(), error_node.assume_init()))
    }

    /// Updates an instantiated graph to match a new graph definition.
    ///
    /// Returns the update result and optionally an error node if the update failed.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g27a7df53a4a5e4a9c3d4d3b5a8a9c3b0)
    ///
    /// # Safety
    /// graph_exec and graph must be valid
    #[cfg(any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    ))]
    pub unsafe fn exec_update(
        graph_exec: sys::CUgraphExec,
        graph: sys::CUgraph,
    ) -> Result<(sys::CUgraphExecUpdateResult, sys::CUgraphNode), DriverError> {
        let mut result_info = MaybeUninit::<sys::CUgraphExecUpdateResultInfo>::uninit();
        sys::cuGraphExecUpdate_v2(graph_exec, graph, result_info.as_mut_ptr()).result()?;
        let result_info = result_info.assume_init();
        Ok((result_info.result, result_info.errorNode))
    }

    /// Gets the parameters of a kernel node.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5a8a9c3d4d3b5a8a9c3b0d7)
    ///
    /// # Safety
    /// node and node_params must be valid
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn kernel_node_get_params(
        node: sys::CUgraphNode,
        node_params: *mut sys::CUDA_KERNEL_NODE_PARAMS,
    ) -> Result<(), DriverError> {
        sys::cuGraphKernelNodeGetParams(node, node_params).result()
    }

    /// Gets the parameters of a kernel node.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5a8a9c3d4d3b5a8a9c3b0d7)
    ///
    /// # Safety
    /// node and node_params must be valid
    #[cfg(any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010"
    ))]
    pub unsafe fn kernel_node_get_params(
        node: sys::CUgraphNode,
        node_params: *mut sys::CUDA_KERNEL_NODE_PARAMS,
    ) -> Result<(), DriverError> {
        sys::cuGraphKernelNodeGetParams_v2(node, node_params).result()
    }

    /// Gets the parameters of a memcpy node.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6a8a9c3d4d3b5a8a9c3b0d7)
    ///
    /// # Safety
    /// node and node_params must be valid
    pub unsafe fn memcpy_node_get_params(
        node: sys::CUgraphNode,
        node_params: *mut sys::CUDA_MEMCPY3D,
    ) -> Result<(), DriverError> {
        sys::cuGraphMemcpyNodeGetParams(node, node_params).result()
    }
}

pub mod mem_pool {
    //! Memory pool management functions (`cuMemPool*`).
    //!
    //! See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html)

    use super::{
        sys::{self},
        DriverError,
    };
    use std::mem::MaybeUninit;

    /// Creates a memory pool on the specified device.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g4f2c67c59a7adfe6ba65cca1c7b6fd45)
    ///
    /// # Safety
    /// Device must be valid.
    pub unsafe fn create(
        _device: sys::CUdevice,
        ordinal: i32,
    ) -> Result<sys::CUmemoryPool, DriverError> {
        let mut pool = MaybeUninit::uninit();

        // Initialize pool properties with zeros
        let mut props: sys::CUmemPoolProps = std::mem::zeroed();
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
        props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = ordinal;

        sys::cuMemPoolCreate(pool.as_mut_ptr(), &props).result()?;
        Ok(pool.assume_init())
    }

    /// Destroys a memory pool.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g6e3a4b4b4b4b4b4b4b4b4b4b4b4b4b4b)
    ///
    /// # Safety
    /// Pool must be valid and not already destroyed.
    pub unsafe fn destroy(pool: sys::CUmemoryPool) -> Result<(), DriverError> {
        sys::cuMemPoolDestroy(pool).result()
    }

    /// Trim the pool, releasing memory back to the OS.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g6a9a5f1b4f0b7e5c0e6e0a7c9a6a8a8a)
    ///
    /// # Safety
    /// Pool must be valid.
    pub unsafe fn trim(
        pool: sys::CUmemoryPool,
        min_bytes_to_keep: usize,
    ) -> Result<(), DriverError> {
        sys::cuMemPoolTrimTo(pool, min_bytes_to_keep).result()
    }

    /// Get a pool attribute.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f)
    ///
    /// # Safety
    /// Pool must be valid.
    pub unsafe fn get_attribute(
        pool: sys::CUmemoryPool,
        attr: sys::CUmemPool_attribute,
    ) -> Result<u64, DriverError> {
        let mut value: u64 = 0;
        sys::cuMemPoolGetAttribute(pool, attr, &mut value as *mut u64 as *mut _).result()?;
        Ok(value)
    }

    /// Set a pool attribute.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f)
    ///
    /// # Safety
    /// Pool must be valid.
    pub unsafe fn set_attribute(
        pool: sys::CUmemoryPool,
        attr: sys::CUmemPool_attribute,
        value: u64,
    ) -> Result<(), DriverError> {
        let mut val = value;
        sys::cuMemPoolSetAttribute(pool, attr, &mut val as *mut u64 as *mut _).result()
    }

    /// Allocate memory from a pool asynchronously.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f5f)
    ///
    /// # Safety
    /// Pool and stream must be valid.
    pub unsafe fn alloc_async(
        pool: sys::CUmemoryPool,
        size: usize,
        stream: sys::CUstream,
    ) -> Result<sys::CUdeviceptr, DriverError> {
        let mut dev_ptr = MaybeUninit::uninit();
        sys::cuMemAllocFromPoolAsync(dev_ptr.as_mut_ptr(), size, pool, stream).result()?;
        Ok(dev_ptr.assume_init())
    }

    /// Get the default memory pool for a device.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g26aa5b41f58e5cb8f9e5e9b1a3e8e8e8)
    ///
    /// # Safety
    /// Device must be valid.
    pub unsafe fn get_default(device: sys::CUdevice) -> Result<sys::CUmemoryPool, DriverError> {
        let mut pool = MaybeUninit::uninit();
        sys::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), device).result()?;
        Ok(pool.assume_init())
    }

    /// Get the current memory pool for a device.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8d4f6a4b6b9c5d7a0a0a0a0a0a0a0a0a)
    ///
    /// # Safety
    /// Device must be valid.
    pub unsafe fn get_current(device: sys::CUdevice) -> Result<sys::CUmemoryPool, DriverError> {
        let mut pool = MaybeUninit::uninit();
        sys::cuDeviceGetMemPool(pool.as_mut_ptr(), device).result()?;
        Ok(pool.assume_init())
    }

    /// Set the current memory pool for a device.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g0c0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a)
    ///
    /// # Safety
    /// Device and pool must be valid.
    pub unsafe fn set_current(
        device: sys::CUdevice,
        pool: sys::CUmemoryPool,
    ) -> Result<(), DriverError> {
        sys::cuDeviceSetMemPool(device, pool).result()
    }
}

#[cfg(test)]
mod tests {
    use super::super::safe::{CudaContext, CudaSlice};
    use super::*;
    use std::println;

    #[test]
    fn peer_transfer_contexts() -> Result<(), DriverError> {
        let ctx1 = CudaContext::new(0)?;
        if device::get_count()? < 2 {
            println!("Skip test because not enough cuda devices");
            return Ok(());
        }
        let stream1 = ctx1.default_stream();
        let a: CudaSlice<f64> = stream1.alloc_zeros::<f64>(10)?;

        let ctx2 = CudaContext::new(1)?;
        let stream2 = ctx2.default_stream();
        let b = stream2.clone_dtod(&a)?;
        let _ = stream1.clone_dtoh(&a)?;
        let _ = stream2.clone_dtoh(&b)?;
        Ok(())
    }

    #[test]
    fn peer_transfer_self() -> Result<(), DriverError> {
        let ctx1 = CudaContext::new(0)?;
        let stream1 = ctx1.default_stream();
        let a: CudaSlice<f64> = stream1.alloc_zeros::<f64>(10)?;

        let ctx2 = CudaContext::new(0)?;
        let stream2 = ctx2.default_stream();
        let b = stream2.clone_dtod(&a)?;
        let _ = stream1.clone_dtoh(&a)?;
        let _ = stream2.clone_dtoh(&b)?;
        Ok(())
    }

    #[test]
    fn re_associate_context_for_memory_op() -> Result<(), DriverError> {
        let ctx1 = CudaContext::new(0)?;
        if device::get_count()? < 2 {
            println!("Skip test because not enough cuda devices");
            return Ok(());
        }
        let stream1 = ctx1.default_stream();
        let a: CudaSlice<f64> = stream1.alloc_zeros::<f64>(10)?;

        let _ctx2 = CudaContext::new(1)?;

        stream1.clone_dtoh(&a).map(|_| ())
    }
}
