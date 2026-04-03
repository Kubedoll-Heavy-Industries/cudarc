use std::mem::MaybeUninit;

use super::sys;

/// Wrapper around [sys::cusolverStatus_t]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CusolverError(pub sys::cusolverStatus_t);

impl sys::cusolverStatus_t {
    pub fn result(self) -> Result<(), CusolverError> {
        match self {
            sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS => Ok(()),
            _ => Err(CusolverError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CusolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CusolverError {}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdncreate)
pub fn dn_create() -> Result<sys::cusolverDnHandle_t, CusolverError> {
    let mut handle = MaybeUninit::uninit();
    unsafe { sys::cusolverDnCreate(handle.as_mut_ptr()) }.result()?;
    Ok(unsafe { handle.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdndestroy)
///
/// # Safety
/// Make sure `handle` has not already been freed
pub unsafe fn dn_destroy(handle: sys::cusolverDnHandle_t) -> Result<(), CusolverError> {
    sys::cusolverDnDestroy(handle).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnsetstream)
///
/// # Safety
/// Make sure `handle` and `stream` are valid (not destroyed)
pub unsafe fn dn_set_stream(
    handle: sys::cusolverDnHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CusolverError> {
    sys::cusolverDnSetStream(handle, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdngetdeterministicmode)
///
/// # Safety
/// Make sure `handle` is valid (not destroyed)
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000",
))]
pub unsafe fn dn_get_deterministic_mode(
    handle: sys::cusolverDnHandle_t,
) -> Result<sys::cusolverDeterministicMode_t, CusolverError> {
    let mut mode = MaybeUninit::uninit();
    sys::cusolverDnGetDeterministicMode(handle, mode.as_mut_ptr()).result()?;
    Ok(unsafe { mode.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnsetdeterministicmode)
///
/// # Safety
/// Make sure `handle` is valid (not destroyed)
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000",
))]
pub unsafe fn dn_set_deterministic_mode(
    handle: sys::cusolverDnHandle_t,
    mode: sys::cusolverDeterministicMode_t,
) -> Result<(), CusolverError> {
    sys::cusolverDnSetDeterministicMode(handle, mode).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdncreateparams)
pub fn dn_create_params() -> Result<sys::cusolverDnParams_t, CusolverError> {
    let mut params = MaybeUninit::uninit();
    unsafe { sys::cusolverDnCreateParams(params.as_mut_ptr()) }.result()?;
    Ok(unsafe { params.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnsetadvoptions)
///
/// # Safety
/// Make sure `params` is valid (not destroyed)
pub unsafe fn dn_set_adv_options(
    params: sys::cusolverDnParams_t,
    function: sys::cusolverDnFunction_t,
    algo: sys::cusolverAlgMode_t,
) -> Result<(), CusolverError> {
    sys::cusolverDnSetAdvOptions(params, function, algo).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdndestroyparams)
///
/// # Safety
/// Make sure `params` is valid (not destroyed)
pub unsafe fn dn_destroy_params(params: sys::cusolverDnParams_t) -> Result<(), CusolverError> {
    sys::cusolverDnDestroyParams(params).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverspcreate)
pub fn sp_create() -> Result<sys::cusolverSpHandle_t, CusolverError> {
    let mut handle = MaybeUninit::uninit();
    unsafe { sys::cusolverSpCreate(handle.as_mut_ptr()) }.result()?;
    Ok(unsafe { handle.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverspdestroy)
///
/// # Safety
/// Make sure `handle` is valid (not destroyed)
pub unsafe fn sp_destroy(handle: sys::cusolverSpHandle_t) -> Result<(), CusolverError> {
    sys::cusolverSpDestroy(handle).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverspsetstream)
///
/// # Safety
/// Make sure `handle` and `stream` are valid (not destroyed)
pub unsafe fn sp_set_stream(
    handle: sys::cusolverSpHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CusolverError> {
    sys::cusolverSpSetStream(handle, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfcreate)
pub fn rf_create() -> Result<sys::cusolverRfHandle_t, CusolverError> {
    let mut handle = MaybeUninit::uninit();
    unsafe { sys::cusolverRfCreate(handle.as_mut_ptr()) }.result()?;
    Ok(unsafe { handle.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfdestroy)
///
/// # Safety
/// Make sure `handle` is valid (not destroyed)
pub unsafe fn rf_destroy(handle: sys::cusolverRfHandle_t) -> Result<(), CusolverError> {
    sys::cusolverRfDestroy(handle).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetmatrixformat)
pub unsafe fn rf_set_matrix_format(
    handle: sys::cusolverRfHandle_t,
    format: sys::cusolverRfMatrixFormat_t,
    diag: sys::cusolverRfUnitDiagonal_t,
) -> Result<(), CusolverError> {
    sys::cusolverRfSetMatrixFormat(handle, format, diag).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetnumericproperties)
pub unsafe fn rf_set_numeric_properties(
    handle: sys::cusolverRfHandle_t,
    zero: f64,
    boost: f64,
) -> Result<(), CusolverError> {
    sys::cusolverRfSetNumericProperties(handle, zero, boost).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetresetvaluesfastmode)
pub unsafe fn rf_set_reset_values_fast_mode(
    handle: sys::cusolverRfHandle_t,
    fast_mode: sys::cusolverRfResetValuesFastMode_t,
) -> Result<(), CusolverError> {
    sys::cusolverRfSetResetValuesFastMode(handle, fast_mode).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetalgs)
pub unsafe fn rf_set_algs(
    handle: sys::cusolverRfHandle_t,
    fact_alg: sys::cusolverRfFactorization_t,
    alg: sys::cusolverRfTriangularSolve_t,
) -> Result<(), CusolverError> {
    sys::cusolverRfSetAlgs(handle, fact_alg, alg).result()
}

/// GETRF buffer size for f32
pub unsafe fn dn_sgetrf_buffer_size(
    handle: sys::cusolverDnHandle_t,
    m: i32,
    n: i32,
    a: *mut f32,
    lda: i32,
    lwork: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSgetrf_bufferSize(handle, m, n, a, lda, lwork).result()
}

/// GETRF for f32
pub unsafe fn dn_sgetrf(
    handle: sys::cusolverDnHandle_t,
    m: i32,
    n: i32,
    a: *mut f32,
    lda: i32,
    workspace: *mut f32,
    ipiv: *mut i32,
    info: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSgetrf(handle, m, n, a, lda, workspace, ipiv, info).result()
}

/// GETRS for f32
pub unsafe fn dn_sgetrs(
    handle: sys::cusolverDnHandle_t,
    trans: sys::cublasOperation_t,
    n: i32,
    nrhs: i32,
    a: *const f32,
    lda: i32,
    ipiv: *const i32,
    b: *mut f32,
    ldb: i32,
    info: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSgetrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info).result()
}

/// POTRF buffer size for f32
pub unsafe fn dn_spotrf_buffer_size(
    handle: sys::cusolverDnHandle_t,
    uplo: sys::cublasFillMode_t,
    n: i32,
    a: *mut f32,
    lda: i32,
    lwork: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSpotrf_bufferSize(handle, uplo, n, a, lda, lwork).result()
}

/// POTRF for f32
pub unsafe fn dn_spotrf(
    handle: sys::cusolverDnHandle_t,
    uplo: sys::cublasFillMode_t,
    n: i32,
    a: *mut f32,
    lda: i32,
    workspace: *mut f32,
    lwork: i32,
    info: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSpotrf(handle, uplo, n, a, lda, workspace, lwork, info).result()
}

/// GEQRF buffer size for f32
pub unsafe fn dn_sgeqrf_buffer_size(
    handle: sys::cusolverDnHandle_t,
    m: i32,
    n: i32,
    a: *mut f32,
    lda: i32,
    lwork: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSgeqrf_bufferSize(handle, m, n, a, lda, lwork).result()
}

/// GEQRF for f32
pub unsafe fn dn_sgeqrf(
    handle: sys::cusolverDnHandle_t,
    m: i32,
    n: i32,
    a: *mut f32,
    lda: i32,
    tau: *mut f32,
    workspace: *mut f32,
    lwork: i32,
    info: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSgeqrf(handle, m, n, a, lda, tau, workspace, lwork, info).result()
}

/// ORGQR buffer size for f32
pub unsafe fn dn_sorgqr_buffer_size(
    handle: sys::cusolverDnHandle_t,
    m: i32,
    n: i32,
    k: i32,
    a: *const f32,
    lda: i32,
    tau: *const f32,
    lwork: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSorgqr_bufferSize(handle, m, n, k, a, lda, tau, lwork).result()
}

/// ORGQR for f32
pub unsafe fn dn_sorgqr(
    handle: sys::cusolverDnHandle_t,
    m: i32,
    n: i32,
    k: i32,
    a: *mut f32,
    lda: i32,
    tau: *const f32,
    work: *mut f32,
    lwork: i32,
    info: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSorgqr(handle, m, n, k, a, lda, tau, work, lwork, info).result()
}

/// GESVD buffer size for f32
pub unsafe fn dn_sgesvd_buffer_size(
    handle: sys::cusolverDnHandle_t,
    m: i32,
    n: i32,
    lwork: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSgesvd_bufferSize(handle, m, n, lwork).result()
}

/// GESVD for f32
pub unsafe fn dn_sgesvd(
    handle: sys::cusolverDnHandle_t,
    jobu: i8,
    jobvt: i8,
    m: i32,
    n: i32,
    a: *mut f32,
    lda: i32,
    s: *mut f32,
    u: *mut f32,
    ldu: i32,
    vt: *mut f32,
    ldvt: i32,
    work: *mut f32,
    lwork: i32,
    rwork: *mut f32,
    info: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSgesvd(
        handle, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info,
    )
    .result()
}

/// SYEVD buffer size for f32
pub unsafe fn dn_ssyevd_buffer_size(
    handle: sys::cusolverDnHandle_t,
    jobz: sys::cusolverEigMode_t,
    uplo: sys::cublasFillMode_t,
    n: i32,
    a: *const f32,
    lda: i32,
    w: *const f32,
    lwork: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, a, lda, w, lwork).result()
}

/// SYEVD for f32
pub unsafe fn dn_ssyevd(
    handle: sys::cusolverDnHandle_t,
    jobz: sys::cusolverEigMode_t,
    uplo: sys::cublasFillMode_t,
    n: i32,
    a: *mut f32,
    lda: i32,
    w: *mut f32,
    work: *mut f32,
    lwork: i32,
    info: *mut i32,
) -> Result<(), CusolverError> {
    sys::cusolverDnSsyevd(handle, jobz, uplo, n, a, lda, w, work, lwork, info).result()
}

#[cfg(test)]
mod tests {
    use super::sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
    use crate::cusolver::result::{
        dn_create, dn_destroy, dn_sgeqrf, dn_sgesvd, dn_sgetrf, dn_spotrf, dn_spotrf_buffer_size,
    };
    use crate::driver::safe::core::DevicePtr;
    use crate::driver::{CudaContext, CudaSlice, CudaStream};

    fn get_ptr<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> *mut T {
        slice.device_ptr(stream).0 as *mut T
    }

    #[test]
    fn test_spotrf_lower() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let n: i32 = 3;
        let a: Vec<f32> = vec![4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 1.0, 2.0, 6.0];
        let mut a_dev = stream.clone_htod(&a).unwrap();
        let lda = n;

        let mut lwork: i32 = 0;
        let mut work = stream.alloc_zeros::<f32>(1).unwrap();
        let mut info = stream.alloc_zeros::<i32>(1).unwrap();

        let handle = dn_create().unwrap();
        unsafe {
            dn_spotrf_buffer_size(
                handle,
                CUBLAS_FILL_MODE_LOWER,
                n,
                get_ptr(&a_dev, &stream),
                lda,
                &mut lwork,
            )
            .unwrap();
        }
        work = unsafe { stream.alloc(lwork as usize).unwrap() };

        unsafe {
            dn_spotrf(
                handle,
                CUBLAS_FILL_MODE_LOWER,
                n,
                get_ptr(&mut a_dev, &stream),
                lda,
                get_ptr(&work, &stream),
                lwork,
                get_ptr(&info, &stream),
            )
            .unwrap();
        }

        let a_result = stream.clone_dtoh(&a_dev).unwrap();
        let info_result = stream.clone_dtoh(&info).unwrap();

        assert_eq!(info_result[0], 0);
        for i in 0..(n as usize) {
            assert!(a_result[i * n as usize + i] > 0.0);
        }

        unsafe { dn_destroy(handle).unwrap() };
    }

    #[test]
    fn test_sgetrf() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let n: i32 = 3;
        let a: Vec<f32> = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0];
        let mut a_dev = stream.clone_htod(&a).unwrap();
        let lda = n;

        let mut ipiv = stream.alloc_zeros::<i32>(n as usize).unwrap();
        let mut work = stream.alloc_zeros::<f32>(n as usize * n as usize).unwrap();
        let mut info = stream.alloc_zeros::<i32>(1).unwrap();

        let handle = dn_create().unwrap();

        unsafe {
            dn_sgetrf(
                handle,
                n,
                n,
                get_ptr(&mut a_dev, &stream),
                lda,
                get_ptr(&work, &stream),
                get_ptr(&ipiv, &stream),
                get_ptr(&info, &stream),
            )
            .unwrap();
        }

        let a_result = stream.clone_dtoh(&a_dev).unwrap();
        let info_result = stream.clone_dtoh(&info).unwrap();

        assert_eq!(info_result[0], 0);

        unsafe { dn_destroy(handle).unwrap() };
    }

    #[test]
    fn test_sgeqrf() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let m: i32 = 3;
        let n: i32 = 4;
        let a: Vec<f32> = vec![
            1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0,
        ];
        let mut a_dev = stream.clone_htod(&a).unwrap();
        let lda = m;

        let mut tau = stream.alloc_zeros::<f32>(m.min(n) as usize).unwrap();
        let mut work = stream.alloc_zeros::<f32>((m * n) as usize).unwrap();
        let mut info = stream.alloc_zeros::<i32>(1).unwrap();

        let handle = dn_create().unwrap();

        unsafe {
            dn_sgeqrf(
                handle,
                m,
                n,
                get_ptr(&mut a_dev, &stream),
                lda,
                get_ptr(&tau, &stream),
                get_ptr(&work, &stream),
                m * n,
                get_ptr(&info, &stream),
            )
            .unwrap();
        }

        let info_result = stream.clone_dtoh(&info).unwrap();
        assert_eq!(info_result[0], 0);

        unsafe { dn_destroy(handle).unwrap() };
    }

    // Note: test_sgesvd is skipped because it requires careful buffer sizing
    // that varies by CUDA version. The sys_test.rs has a working implementation.
}
