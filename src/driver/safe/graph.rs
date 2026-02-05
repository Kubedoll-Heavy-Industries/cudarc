use std::marker::PhantomData;
use std::sync::Arc;

use crate::driver::{result, sys};

use super::{CudaContext, CudaStream, DriverError};

/// Represents a replay-able Cuda Graph. Create with [CudaStream::begin_capture()] and [CudaStream::end_capture()].
///
/// Once created you can replay with [CudaGraph::launch()].
///
/// # On Thread safety
///
/// This object is **NOT** thread safe.
///
/// From official docs:
///
/// > Graph objects (cudaGraph_t, CUgraph) are not internally synchronized and must not be accessed concurrently from multiple threads. API calls accessing the same graph object must be serialized externally.
/// >
/// > Note that this includes APIs which may appear to be read-only, such as cudaGraphClone() (cuGraphClone()) and cudaGraphInstantiate() (cuGraphInstantiate()). No API or pair of APIs is guaranteed to be safe to call on the same graph object from two different threads without serialization.
///
/// <https://docs.nvidia.com/cuda/cuda-driver-api/graphs-thread-safety.html#graphs-thread-safety>
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        let ctx = &self.stream.ctx;

        let cu_graph_exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !cu_graph_exec.is_null() {
            ctx.record_err(unsafe { result::graph::exec_destroy(cu_graph_exec) });
        }

        let cu_graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !cu_graph.is_null() {
            ctx.record_err(unsafe { result::graph::destroy(cu_graph) });
        }
    }
}

impl CudaStream {
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143)
    pub fn begin_capture(&self, mode: sys::CUstreamCaptureMode) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::begin_capture(self.cu_stream, mode) }
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c)
    ///
    /// `flags` is passed to [cuGraphInstantiate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1)
    pub fn end_capture(
        self: &Arc<Self>,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<Option<CudaGraph>, DriverError> {
        self.ctx.bind_to_thread()?;
        let cu_graph = unsafe { result::stream::end_capture(self.cu_stream) }?;
        if cu_graph.is_null() {
            return Ok(None);
        }
        let cu_graph_exec = unsafe { result::graph::instantiate(cu_graph, flags) }?;
        Ok(Some(CudaGraph {
            cu_graph,
            cu_graph_exec,
            stream: self.clone(),
        }))
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca)
    pub fn capture_status(&self) -> Result<sys::CUstreamCaptureStatus, DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::is_capturing(self.cu_stream) }
    }

    /// End capture and return raw graph definition (not instantiated).
    ///
    /// Use this when you need to inspect or modify the graph before instantiation,
    /// or when you want to create multiple executables from one definition.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c)
    pub fn end_capture_graph(self: &Arc<Self>) -> Result<Option<CudaGraphDef>, DriverError> {
        self.ctx.bind_to_thread()?;
        let cu_graph = unsafe { result::stream::end_capture(self.cu_stream) }?;
        if cu_graph.is_null() {
            return Ok(None);
        }
        Ok(Some(CudaGraphDef {
            cu_graph,
            ctx: self.ctx.clone(),
        }))
    }

    /// Check if stream is currently capturing.
    ///
    /// Returns `true` if the stream is in active capture mode, `false` otherwise.
    pub fn is_capturing(&self) -> Result<bool, DriverError> {
        let status = self.capture_status()?;
        Ok(status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
    }

    /// Get detailed capture information.
    ///
    /// Returns the capture status and unique capture sequence ID.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g9d22e54a0755b3b0e01dca4c9a9e70c8)
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
    pub fn capture_info(&self) -> Result<result::stream::CaptureInfo, DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::get_capture_info(self.cu_stream) }
    }
}

impl CudaGraph {
    /// Launches the graph on its capture stream.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471)
    pub fn launch(&self) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;
        unsafe { result::graph::launch(self.cu_graph_exec, self.stream.cu_stream) }
    }

    /// Pre-uploads the graph's resources to the device so that the
    /// first [CudaGraph::launch()] does not incur setup overhead.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gdb81438b083d42a26693f6f2bce150cd)
    pub fn upload(&self) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;
        unsafe { result::graph::upload(self.cu_graph_exec, self.stream.cu_stream) }
    }

    /// Get the underlying [sys::CUgraph].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not destroy the graph**, as it is still
    /// owned by the [CudaGraph].
    pub fn cu_graph(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// Launches the graph on a specific stream.
    ///
    /// The stream must belong to the same context as the graph.
    pub fn launch_on(&self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.stream.ctx != stream.ctx {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.stream.ctx.bind_to_thread()?;
        unsafe { result::graph::launch(self.cu_graph_exec, stream.cu_stream) }
    }

    /// Returns the stream this graph was captured from.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Returns the context this graph belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.stream.ctx
    }

    /// Returns the underlying `CUgraph` definition handle.
    ///
    /// # Safety
    /// Do not destroy this handle.
    pub fn cu_graph(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// Returns the underlying `CUgraphExec` handle.
    ///
    /// # Safety
    /// Do not destroy this handle.
    pub fn cu_graph_exec(&self) -> sys::CUgraphExec {
        self.cu_graph_exec
    }

    /// Returns all nodes in the graph.
    ///
    /// The returned nodes can be used with [CudaGraph::update_kernel_node_params]
    /// and related methods.
    pub fn nodes(&self) -> Result<Vec<CudaGraphNode<'_>>, DriverError> {
        self.stream.ctx.bind_to_thread()?;
        let raw_nodes = unsafe { result::graph::get_nodes(self.cu_graph) }?;
        Ok(raw_nodes
            .into_iter()
            .map(|cu_node| CudaGraphNode {
                cu_node,
                _marker: PhantomData,
            })
            .collect())
    }

    /// Returns root nodes in the graph (nodes with no dependencies).
    pub fn root_nodes(&self) -> Result<Vec<CudaGraphNode<'_>>, DriverError> {
        self.stream.ctx.bind_to_thread()?;
        let raw_nodes = unsafe { result::graph::get_root_nodes(self.cu_graph) }?;
        Ok(raw_nodes
            .into_iter()
            .map(|cu_node| CudaGraphNode {
                cu_node,
                _marker: PhantomData,
            })
            .collect())
    }

    /// Updates the parameters of a kernel node in this graph.
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    /// - The node must be a kernel node from this graph.
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn update_kernel_node_params(
        &mut self,
        node: &CudaGraphNode<'_>,
        params: &KernelNodeParams,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;

        let kernel_params = sys::CUDA_KERNEL_NODE_PARAMS {
            func: params.func,
            gridDimX: params.grid_dim.0,
            gridDimY: params.grid_dim.1,
            gridDimZ: params.grid_dim.2,
            blockDimX: params.block_dim.0,
            blockDimY: params.block_dim.1,
            blockDimZ: params.block_dim.2,
            sharedMemBytes: params.shared_mem_bytes,
            kernelParams: args.as_mut_ptr(),
            extra: std::ptr::null_mut(),
        };

        result::graph::exec_kernel_node_set_params(self.cu_graph_exec, node.cu_node, &kernel_params)
    }

    /// Updates the parameters of a kernel node in this graph.
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    /// - The node must be a kernel node from this graph.
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
    pub unsafe fn update_kernel_node_params(
        &mut self,
        node: &CudaGraphNode<'_>,
        params: &KernelNodeParams,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;

        let kernel_params = sys::CUDA_KERNEL_NODE_PARAMS {
            func: params.func,
            gridDimX: params.grid_dim.0,
            gridDimY: params.grid_dim.1,
            gridDimZ: params.grid_dim.2,
            blockDimX: params.block_dim.0,
            blockDimY: params.block_dim.1,
            blockDimZ: params.block_dim.2,
            sharedMemBytes: params.shared_mem_bytes,
            kernelParams: args.as_mut_ptr(),
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: self.stream.ctx.cu_ctx(),
        };

        result::graph::exec_kernel_node_set_params(self.cu_graph_exec, node.cu_node, &kernel_params)
    }

    /// Updates only the kernel arguments of a kernel node.
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    /// - The node must be a kernel node from this graph.
    pub unsafe fn update_kernel_node_args(
        &mut self,
        node: &CudaGraphNode<'_>,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;

        // Get current parameters
        let mut current_params = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::uninit();
        result::graph::kernel_node_get_params(node.cu_node, current_params.as_mut_ptr())?;
        let mut current_params = current_params.assume_init();

        // Update only the kernel args
        current_params.kernelParams = args.as_mut_ptr();

        result::graph::exec_kernel_node_set_params(
            self.cu_graph_exec,
            node.cu_node,
            &current_params,
        )
    }

    /// Updates a memcpy node's source and destination pointers.
    ///
    /// # Safety
    ///
    /// - `dst` and `src` must be valid device pointers.
    /// - The memory regions must remain valid until graph execution completes.
    /// - The node must be a memcpy node from this graph.
    pub unsafe fn update_memcpy_node_params(
        &mut self,
        node: &CudaGraphNode<'_>,
        dst: sys::CUdeviceptr,
        src: sys::CUdeviceptr,
        size: usize,
    ) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;

        let copy_params = sys::CUDA_MEMCPY3D_st {
            srcXInBytes: 0,
            srcY: 0,
            srcZ: 0,
            srcLOD: 0,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: src,
            srcArray: std::ptr::null_mut(),
            reserved0: std::ptr::null_mut(),
            srcPitch: size,
            srcHeight: 1,
            dstXInBytes: 0,
            dstY: 0,
            dstZ: 0,
            dstLOD: 0,
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst,
            dstArray: std::ptr::null_mut(),
            reserved1: std::ptr::null_mut(),
            dstPitch: size,
            dstHeight: 1,
            WidthInBytes: size,
            Height: 1,
            Depth: 1,
        };

        result::graph::exec_memcpy_node_set_params(
            self.cu_graph_exec,
            node.cu_node,
            &copy_params,
            self.stream.ctx.cu_ctx(),
        )
    }
}

/// A handle to a node within a CUDA graph.
///
/// Lifetime-tracked to prevent use with wrong graph at compile time.
#[derive(Clone, Copy, Debug)]
pub struct CudaGraphNode<'graph> {
    pub(crate) cu_node: sys::CUgraphNode,
    pub(crate) _marker: PhantomData<&'graph CudaGraphDef>,
}

unsafe impl Send for CudaGraphNode<'_> {}
unsafe impl Sync for CudaGraphNode<'_> {}

impl<'graph> CudaGraphNode<'graph> {
    /// Returns the type of this graph node.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g65be75993be27f5c46ee30a3d62203c2)
    pub fn node_type(&self) -> Result<sys::CUgraphNodeType, DriverError> {
        unsafe { result::graph::node_get_type(self.cu_node) }
    }

    /// Returns the underlying `CUgraphNode` handle.
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/destroy the node**, as it is still
    /// owned by the parent graph.
    pub fn cu_node(&self) -> sys::CUgraphNode {
        self.cu_node
    }
}

/// A CUDA graph definition - the template that can be inspected and instantiated.
///
/// This represents the graph structure before instantiation. It can be queried for
/// nodes and edges, cloned, and instantiated into an executable graph.
///
/// # On Thread safety
///
/// This object is **NOT** thread safe.
///
/// From official docs:
///
/// > Graph objects (cudaGraph_t, CUgraph) are not internally synchronized and must not be accessed concurrently from multiple threads. API calls accessing the same graph object must be serialized externally.
/// >
/// > Note that this includes APIs which may appear to be read-only, such as cudaGraphClone() (cuGraphClone()) and cudaGraphInstantiate() (cuGraphInstantiate()). No API or pair of APIs is guaranteed to be safe to call on the same graph object from two different threads without serialization.
///
/// <https://docs.nvidia.com/cuda/cuda-driver-api/graphs-thread-safety.html#graphs-thread-safety>
pub struct CudaGraphDef {
    pub(crate) cu_graph: sys::CUgraph,
    pub(crate) ctx: Arc<CudaContext>,
}

impl Drop for CudaGraphDef {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        let cu_graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !cu_graph.is_null() {
            self.ctx
                .record_err(unsafe { result::graph::destroy(cu_graph) });
        }
    }
}

impl CudaGraphDef {
    /// Returns all nodes in the graph.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g048f6e36f5d7e0ad5f6e2ab38ee37e55)
    pub fn nodes(&self) -> Result<Vec<CudaGraphNode<'_>>, DriverError> {
        self.ctx.bind_to_thread()?;
        let raw_nodes = unsafe { result::graph::get_nodes(self.cu_graph) }?;
        Ok(raw_nodes
            .into_iter()
            .map(|cu_node| CudaGraphNode {
                cu_node,
                _marker: PhantomData,
            })
            .collect())
    }

    /// Returns all root nodes in the graph (nodes with no dependencies).
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00216ee8e72ca27c85c27e3e81e837f6)
    pub fn root_nodes(&self) -> Result<Vec<CudaGraphNode<'_>>, DriverError> {
        self.ctx.bind_to_thread()?;
        let raw_nodes = unsafe { result::graph::get_root_nodes(self.cu_graph) }?;
        Ok(raw_nodes
            .into_iter()
            .map(|cu_node| CudaGraphNode {
                cu_node,
                _marker: PhantomData,
            })
            .collect())
    }

    /// Returns all edges in the graph as (from, to) pairs.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge9d27a6b2ebca4d9e5f94c0c8c8b0e06)
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
    pub fn edges(&self) -> Result<Vec<(CudaGraphNode<'_>, CudaGraphNode<'_>)>, DriverError> {
        self.ctx.bind_to_thread()?;
        let raw_edges = unsafe { result::graph::get_edges(self.cu_graph) }?;
        Ok(raw_edges
            .into_iter()
            .map(|(from, to)| {
                (
                    CudaGraphNode {
                        cu_node: from,
                        _marker: PhantomData,
                    },
                    CudaGraphNode {
                        cu_node: to,
                        _marker: PhantomData,
                    },
                )
            })
            .collect())
    }

    /// Instantiates the graph for execution.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1)
    pub fn instantiate(
        &self,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<CudaGraphExec<'_>, DriverError> {
        self.ctx.bind_to_thread()?;
        let cu_graph_exec = unsafe { result::graph::instantiate(self.cu_graph, flags) }?;
        Ok(CudaGraphExec {
            cu_graph_exec,
            ctx: &self.ctx,
            _marker: PhantomData,
        })
    }

    /// Creates a clone of this graph.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g9d5cfeb00b8ee918ea3c6f0816b4d8ef)
    pub fn try_clone(&self) -> Result<Self, DriverError> {
        self.ctx.bind_to_thread()?;
        let cloned_graph = unsafe { result::graph::clone(self.cu_graph) }?;
        Ok(CudaGraphDef {
            cu_graph: cloned_graph,
            ctx: self.ctx.clone(),
        })
    }

    /// Returns the underlying `CUgraph` handle.
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/destroy the graph**, as it is still
    /// owned by this [CudaGraphDef].
    pub fn cu_graph(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// Returns a reference to the CUDA context this graph was created in.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

/// An executable CUDA graph that can be launched on a stream.
///
/// Created by calling [CudaGraphDef::instantiate].
///
/// # On Thread safety
///
/// This object is **NOT** thread safe.
///
/// From official docs:
///
/// > Executable graph objects (cudaGraphExec_t, CUgraphExec) are not internally synchronized and must not be accessed concurrently from multiple threads. API calls accessing the same cudaGraphExec_t must be serialized externally.
///
/// <https://docs.nvidia.com/cuda/cuda-driver-api/graphs-thread-safety.html#graphs-thread-safety>
pub struct CudaGraphExec<'def> {
    pub(crate) cu_graph_exec: sys::CUgraphExec,
    pub(crate) ctx: &'def Arc<CudaContext>,
    pub(crate) _marker: PhantomData<&'def CudaGraphDef>,
}

impl Drop for CudaGraphExec<'_> {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        let cu_graph_exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !cu_graph_exec.is_null() {
            self.ctx
                .record_err(unsafe { result::graph::exec_destroy(cu_graph_exec) });
        }
    }
}

/// Parameters for updating a kernel node in an instantiated graph.
#[derive(Debug, Clone)]
pub struct KernelNodeParams {
    /// The kernel function to execute
    pub func: sys::CUfunction,
    /// Grid dimensions (number of blocks in each dimension)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (number of threads per block in each dimension)
    pub block_dim: (u32, u32, u32),
    /// Amount of dynamic shared memory to allocate
    pub shared_mem_bytes: u32,
}

/// Result of updating an instantiated graph from a modified graph definition.
#[derive(Debug)]
pub struct GraphUpdateResult {
    /// The update result code
    pub result: sys::CUgraphExecUpdateResult,
    /// If the update failed, this may contain the node that caused the error.
    /// Note: This is a raw handle and may be from a different graph.
    pub error_node: Option<sys::CUgraphNode>,
}

impl GraphUpdateResult {
    /// Returns `true` if the update was successful.
    pub fn is_success(&self) -> bool {
        self.result == sys::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS
    }
}

impl<'def> CudaGraphExec<'def> {
    /// Launches this executable graph on the given stream.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471)
    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.ctx != &stream.ctx {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.ctx.bind_to_thread()?;
        unsafe { result::graph::launch(self.cu_graph_exec, stream.cu_stream) }
    }

    /// Updates the parameters of a kernel node in this executable graph.
    ///
    /// This allows changing kernel function, grid/block dimensions, shared memory,
    /// and kernel arguments without re-instantiating the graph.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c)
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    /// - The node must be a kernel node from the graph that was used to create this executable.
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn set_kernel_node_params(
        &mut self,
        node: &CudaGraphNode<'def>,
        params: &KernelNodeParams,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;

        let kernel_params = sys::CUDA_KERNEL_NODE_PARAMS {
            func: params.func,
            gridDimX: params.grid_dim.0,
            gridDimY: params.grid_dim.1,
            gridDimZ: params.grid_dim.2,
            blockDimX: params.block_dim.0,
            blockDimY: params.block_dim.1,
            blockDimZ: params.block_dim.2,
            sharedMemBytes: params.shared_mem_bytes,
            kernelParams: args.as_mut_ptr(),
            extra: std::ptr::null_mut(),
        };

        result::graph::exec_kernel_node_set_params(self.cu_graph_exec, node.cu_node, &kernel_params)
    }

    /// Updates the parameters of a kernel node in this executable graph.
    ///
    /// This allows changing kernel function, grid/block dimensions, shared memory,
    /// and kernel arguments without re-instantiating the graph.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c)
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    /// - The node must be a kernel node from the graph that was used to create this executable.
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
    pub unsafe fn set_kernel_node_params(
        &mut self,
        node: &CudaGraphNode<'def>,
        params: &KernelNodeParams,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;

        let kernel_params = sys::CUDA_KERNEL_NODE_PARAMS {
            func: params.func,
            gridDimX: params.grid_dim.0,
            gridDimY: params.grid_dim.1,
            gridDimZ: params.grid_dim.2,
            blockDimX: params.block_dim.0,
            blockDimY: params.block_dim.1,
            blockDimZ: params.block_dim.2,
            sharedMemBytes: params.shared_mem_bytes,
            kernelParams: args.as_mut_ptr(),
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: self.ctx.cu_ctx(),
        };

        result::graph::exec_kernel_node_set_params(self.cu_graph_exec, node.cu_node, &kernel_params)
    }

    /// Updates only the kernel arguments of a kernel node.
    ///
    /// This is a convenience wrapper that fetches the current node parameters
    /// and updates only the kernel arguments, preserving other settings.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c)
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    /// - The node must be a kernel node from the graph that was used to create this executable.
    pub unsafe fn set_kernel_node_args(
        &mut self,
        node: &CudaGraphNode<'def>,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;

        // Get current parameters
        let mut current_params = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::uninit();
        result::graph::kernel_node_get_params(node.cu_node, current_params.as_mut_ptr())?;
        let mut current_params = current_params.assume_init();

        // Update only the kernel args
        current_params.kernelParams = args.as_mut_ptr();

        result::graph::exec_kernel_node_set_params(
            self.cu_graph_exec,
            node.cu_node,
            &current_params,
        )
    }

    /// Updates the parameters of a memcpy node in this executable graph.
    ///
    /// This is a simplified interface for device-to-device memcpy updates.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50a5c0a1a5a6b0c7b3e3d5a8a9c3b0d7)
    ///
    /// # Safety
    ///
    /// - `dst` and `src` must be valid device pointers.
    /// - The memory regions must remain valid until graph execution completes.
    /// - The node must be a memcpy node from the graph that was used to create this executable.
    pub unsafe fn set_memcpy_node_params(
        &mut self,
        node: &CudaGraphNode<'def>,
        dst: sys::CUdeviceptr,
        src: sys::CUdeviceptr,
        size: usize,
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;

        // Create a 1D memcpy descriptor
        let copy_params = sys::CUDA_MEMCPY3D_st {
            srcXInBytes: 0,
            srcY: 0,
            srcZ: 0,
            srcLOD: 0,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: src,
            srcArray: std::ptr::null_mut(),
            reserved0: std::ptr::null_mut(),
            srcPitch: size,
            srcHeight: 1,
            dstXInBytes: 0,
            dstY: 0,
            dstZ: 0,
            dstLOD: 0,
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst,
            dstArray: std::ptr::null_mut(),
            reserved1: std::ptr::null_mut(),
            dstPitch: size,
            dstHeight: 1,
            WidthInBytes: size,
            Height: 1,
            Depth: 1,
        };

        result::graph::exec_memcpy_node_set_params(
            self.cu_graph_exec,
            node.cu_node,
            &copy_params,
            self.ctx.cu_ctx(),
        )
    }

    /// Updates this executable graph to match a modified graph definition.
    ///
    /// If the topology matches, parameters are updated in-place. This is more
    /// efficient than destroying and re-instantiating the executable graph.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g27a7df53a4a5e4a9c3d4d3b5a8a9c3b0)
    pub fn update(&mut self, graph: &CudaGraphDef) -> Result<GraphUpdateResult, DriverError> {
        self.ctx.bind_to_thread()?;
        let (result, error_node) =
            unsafe { result::graph::exec_update(self.cu_graph_exec, graph.cu_graph) }?;
        Ok(GraphUpdateResult {
            result,
            error_node: if error_node.is_null() {
                None
            } else {
                Some(error_node)
            },
        })
    }

    /// Returns the underlying `CUgraphExec` handle.
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not free/destroy the exec handle**, as it is still
    /// owned by this [CudaGraphExec].
    pub fn cu_graph_exec(&self) -> sys::CUgraphExec {
        self.cu_graph_exec
    }
}
