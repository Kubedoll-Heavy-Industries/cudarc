#[cfg(test)]
mod tests {
    use crate::cutensor::safe::{CuTensor, TensorDescriptor};
    use crate::cutensor::sys::cudaDataType_t;
    use crate::driver::CudaContext;

    #[test]
    fn test_handle_creation() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let cutensor = CuTensor::new(stream.clone()).unwrap();
        let version = cutensor.version();
        println!(
            "cuTENSOR version: {}.{}.{}",
            version.0, version.1, version.2
        );

        assert!(version.0 > 0);
    }

    #[test]
    fn test_tensor_descriptor_creation() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let cutensor = CuTensor::new(stream.clone()).unwrap();

        let extent = vec![4i64, 4];
        let desc = cutensor
            .create_tensor_descriptor(&extent, None, cudaDataType_t::CUDA_R_32F)
            .unwrap();

        assert_eq!(desc.num_modes(), 2);
    }

    #[test]
    fn test_tensor_descriptor_with_strides() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let cutensor = CuTensor::new(stream.clone()).unwrap();

        let extent = vec![3i64, 4, 5];
        let stride = vec![20, 5, 1];
        let desc = cutensor
            .create_tensor_descriptor(&extent, Some(&stride), cudaDataType_t::CUDA_R_32F)
            .unwrap();

        assert_eq!(desc.num_modes(), 3);
    }

    #[test]
    fn test_tensor_descriptor_different_types() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let cutensor = CuTensor::new(stream.clone()).unwrap();

        let extent = vec![2i64, 2];

        let desc_f32 = cutensor
            .create_tensor_descriptor(&extent, None, cudaDataType_t::CUDA_R_32F)
            .unwrap();

        let desc_f16 = cutensor
            .create_tensor_descriptor(&extent, None, cudaDataType_t::CUDA_R_16F)
            .unwrap();

        let desc_i32 = cutensor
            .create_tensor_descriptor(&extent, None, cudaDataType_t::CUDA_R_32I)
            .unwrap();

        assert!(desc_f32.num_modes() > 0);
        assert!(desc_f16.num_modes() > 0);
        assert!(desc_i32.num_modes() > 0);
    }
}
