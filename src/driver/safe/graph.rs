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
#[deprecated(
    since = "0.13.0",
    note = "Use CudaGraphDef and CudaGraphExec for type-safe graph management"
)]
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
    // Prevent auto-impl of Send/Sync - CUDA graphs are NOT thread-safe
    _not_send_sync: PhantomData<*const ()>,
}

#[allow(deprecated)]
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

#[allow(deprecated)]
impl std::fmt::Debug for CudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraph")
            .field("cu_graph", &self.cu_graph)
            .field("cu_graph_exec", &self.cu_graph_exec)
            .finish_non_exhaustive()
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
    #[allow(deprecated)]
    pub fn end_capture(
        self: &Arc<Self>,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<Option<CudaGraph>, DriverError> {
        self.ctx.bind_to_thread()?;
        let cu_graph = unsafe { result::stream::end_capture(self.cu_stream) }?;
        if cu_graph.is_null() {
            return Ok(None);
        }
        // Clean up the graph if instantiation fails to prevent resource leak
        let cu_graph_exec = match unsafe { result::graph::instantiate(cu_graph, flags) } {
            Ok(exec) => exec,
            Err(e) => {
                let _ = unsafe { result::graph::destroy(cu_graph) };
                return Err(e);
            }
        };
        Ok(Some(CudaGraph {
            cu_graph,
            cu_graph_exec,
            stream: self.clone(),
            _not_send_sync: PhantomData,
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
            _not_send_sync: PhantomData,
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
    #[cfg(cuda_11_4_plus)]
    pub fn capture_info(&self) -> Result<result::stream::CaptureInfo, DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::get_capture_info(self.cu_stream) }
    }
}

#[allow(deprecated)]
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
    #[inline]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Returns the context this graph belongs to.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.stream.ctx
    }

    /// Returns the underlying `CUgraph` definition handle.
    ///
    /// # Safety
    /// Do not destroy this handle.
    #[inline]
    pub fn cu_graph(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// Returns the underlying `CUgraphExec` handle.
    ///
    /// # Safety
    /// Do not destroy this handle.
    #[inline]
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
    #[cfg(cuda_11_only)]
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
    #[cfg(cuda_12_plus)]
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

// Note: CudaGraphNode is intentionally NOT Send/Sync because CUDA graphs
// are not thread-safe. The node holds a raw handle that could become invalid
// if the parent graph is modified/destroyed on another thread.

impl<'graph> CudaGraphNode<'graph> {
    /// Returns the type of this graph node.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g65be75993be27f5c46ee30a3d62203c2)
    #[inline]
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
    #[inline]
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
    // Prevent auto-impl of Send/Sync - CUDA graphs are NOT thread-safe
    _not_send_sync: PhantomData<*const ()>,
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

impl std::fmt::Debug for CudaGraphDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraphDef")
            .field("cu_graph", &self.cu_graph)
            .finish_non_exhaustive()
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
    #[cfg(cuda_11_4_plus)]
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
    /// The returned [CudaGraphExec] is lifetime-bound to this graph definition,
    /// ensuring that node handles used for parameter updates always come from
    /// this graph.
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
            ctx: self.ctx.clone(),
            _marker: PhantomData,
        })
    }

    /// Instantiate the graph with raw flags (u64). Use this when the flags value
    /// is 0 (no flags), since the `CUgraphInstantiate_flags` enum has no 0 variant.
    pub fn instantiate_raw(
        &self,
        flags: u64,
    ) -> Result<CudaGraphExec<'_>, DriverError> {
        self.ctx.bind_to_thread()?;
        let cu_graph_exec =
            unsafe { result::graph::instantiate_raw(self.cu_graph, flags) }?;
        Ok(CudaGraphExec {
            cu_graph_exec,
            ctx: self.ctx.clone(),
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
            _not_send_sync: PhantomData,
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
    #[inline]
    pub fn cu_graph(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// Returns a reference to the CUDA context this graph was created in.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Creates an empty graph definition.
    ///
    /// This allows building graphs programmatically by adding nodes with
    /// [`add_empty_node`](Self::add_empty_node), [`add_kernel_node`](Self::add_kernel_node),
    /// [`add_memcpy_node`](Self::add_memcpy_node), etc.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf)
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let cu_graph = unsafe { result::graph::create(0) }?;
        Ok(CudaGraphDef {
            cu_graph,
            ctx: ctx.clone(),
            _not_send_sync: PhantomData,
        })
    }

    /// Adds an empty node to the graph.
    ///
    /// Empty nodes are useful as synchronization points in the graph.
    /// They can have dependencies and be depended upon, but perform no work.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4e0f5c93ce77f3f99e14fd1a00ce8c08)
    pub fn add_empty_node(
        &self,
        dependencies: &[CudaGraphNode<'_>],
    ) -> Result<CudaGraphNode<'_>, DriverError> {
        self.ctx.bind_to_thread()?;

        let dep_ptrs: Vec<sys::CUgraphNode> = dependencies.iter().map(|n| n.cu_node).collect();
        let dep_ptr = if dep_ptrs.is_empty() {
            std::ptr::null()
        } else {
            dep_ptrs.as_ptr()
        };

        let cu_node =
            unsafe { result::graph::add_empty_node(self.cu_graph, dep_ptr, dep_ptrs.len()) }?;

        Ok(CudaGraphNode {
            cu_node,
            _marker: PhantomData,
        })
    }

    /// Adds a kernel node to the graph.
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    /// - The kernel function must be valid for the graph's context.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b0c32e0e8ced67db5d)
    #[cfg(cuda_11_only)]
    pub unsafe fn add_kernel_node(
        &self,
        params: &KernelNodeParams,
        args: &mut [*mut std::ffi::c_void],
        dependencies: &[CudaGraphNode<'_>],
    ) -> Result<CudaGraphNode<'_>, DriverError> {
        self.ctx.bind_to_thread()?;

        let dep_ptrs: Vec<sys::CUgraphNode> = dependencies.iter().map(|n| n.cu_node).collect();
        let dep_ptr = if dep_ptrs.is_empty() {
            std::ptr::null()
        } else {
            dep_ptrs.as_ptr()
        };

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

        let cu_node = result::graph::add_kernel_node(
            self.cu_graph,
            dep_ptr,
            dep_ptrs.len(),
            &kernel_params,
        )?;

        Ok(CudaGraphNode {
            cu_node,
            _marker: PhantomData,
        })
    }

    /// Adds a kernel node to the graph.
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    /// - The kernel function must be valid for the graph's context.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b0c32e0e8ced67db5d)
    #[cfg(cuda_12_plus)]
    pub unsafe fn add_kernel_node(
        &self,
        params: &KernelNodeParams,
        args: &mut [*mut std::ffi::c_void],
        dependencies: &[CudaGraphNode<'_>],
    ) -> Result<CudaGraphNode<'_>, DriverError> {
        self.ctx.bind_to_thread()?;

        let dep_ptrs: Vec<sys::CUgraphNode> = dependencies.iter().map(|n| n.cu_node).collect();
        let dep_ptr = if dep_ptrs.is_empty() {
            std::ptr::null()
        } else {
            dep_ptrs.as_ptr()
        };

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

        let cu_node = result::graph::add_kernel_node(
            self.cu_graph,
            dep_ptr,
            dep_ptrs.len(),
            &kernel_params,
        )?;

        Ok(CudaGraphNode {
            cu_node,
            _marker: PhantomData,
        })
    }

    /// Adds a device-to-device memcpy node to the graph.
    ///
    /// # Safety
    ///
    /// - `dst` and `src` must be valid device pointers.
    /// - The memory regions must remain valid until graph execution completes.
    /// - The memory regions must not overlap.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5f3a)
    pub unsafe fn add_memcpy_node(
        &self,
        dst: sys::CUdeviceptr,
        src: sys::CUdeviceptr,
        size: usize,
        dependencies: &[CudaGraphNode<'_>],
    ) -> Result<CudaGraphNode<'_>, DriverError> {
        self.ctx.bind_to_thread()?;

        let dep_ptrs: Vec<sys::CUgraphNode> = dependencies.iter().map(|n| n.cu_node).collect();
        let dep_ptr = if dep_ptrs.is_empty() {
            std::ptr::null()
        } else {
            dep_ptrs.as_ptr()
        };

        // Create a 1D device-to-device memcpy descriptor
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

        let cu_node = result::graph::add_memcpy_node(
            self.cu_graph,
            dep_ptr,
            dep_ptrs.len(),
            &copy_params,
            self.ctx.cu_ctx(),
        )?;

        Ok(CudaGraphNode {
            cu_node,
            _marker: PhantomData,
        })
    }

    /// Adds dependencies between existing nodes in the graph.
    ///
    /// Each pair `(from[i], to[i])` creates an edge where `from[i]` must
    /// complete before `to[i]` can execute.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g3acf23cfc62a5c4c8e044d21b5c42b6d)
    pub fn add_dependencies(
        &self,
        from: &[CudaGraphNode<'_>],
        to: &[CudaGraphNode<'_>],
    ) -> Result<(), DriverError> {
        if from.len() != to.len() {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_VALUE));
        }
        if from.is_empty() {
            return Ok(());
        }

        self.ctx.bind_to_thread()?;

        let from_ptrs: Vec<sys::CUgraphNode> = from.iter().map(|n| n.cu_node).collect();
        let to_ptrs: Vec<sys::CUgraphNode> = to.iter().map(|n| n.cu_node).collect();

        unsafe {
            result::graph::add_dependencies(
                self.cu_graph,
                from_ptrs.as_ptr(),
                to_ptrs.as_ptr(),
                from_ptrs.len(),
            )
        }
    }
}

/// An executable CUDA graph that can be launched on a stream.
///
/// Created by calling [CudaGraphDef::instantiate].
///
/// The `'graph` lifetime ties this executable to its source [CudaGraphDef]. This ensures
/// that [CudaGraphNode] handles used for parameter updates always come from the correct graph,
/// preventing misuse at compile time:
///
/// ```compile_fail
/// let def_a = stream.end_capture_graph()?.unwrap();
/// let def_b = stream.end_capture_graph()?.unwrap();
/// let mut exec_a = def_a.instantiate(0)?;
/// let nodes_b = def_b.nodes()?;
/// // Error: lifetime mismatch - nodes_b has wrong 'graph lifetime
/// unsafe { exec_a.set_kernel_node_args(&nodes_b[0], &mut args)?; }
/// ```
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
pub struct CudaGraphExec<'graph> {
    pub(crate) cu_graph_exec: sys::CUgraphExec,
    pub(crate) ctx: Arc<CudaContext>,
    // Prevent auto-impl of Send/Sync - CUDA graphs are NOT thread-safe
    // Also tracks the lifetime of the source CudaGraphDef
    _marker: PhantomData<(&'graph CudaGraphDef, *const ())>,
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

impl std::fmt::Debug for CudaGraphExec<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraphExec")
            .field("cu_graph_exec", &self.cu_graph_exec)
            .finish_non_exhaustive()
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
    #[inline]
    pub fn is_success(&self) -> bool {
        self.result == sys::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS
    }
}

impl<'graph> CudaGraphExec<'graph> {
    /// Launches this executable graph on the given stream.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471)
    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.ctx != stream.ctx {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.ctx.bind_to_thread()?;
        unsafe { result::graph::launch(self.cu_graph_exec, stream.cu_stream) }
    }

    /// Returns a reference to the CUDA context this executable was created in.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Updates the parameters of a kernel node in this executable graph.
    ///
    /// This allows changing kernel function, grid/block dimensions, shared memory,
    /// and kernel arguments without re-instantiating the graph.
    ///
    /// The node must come from the same graph this executable was instantiated from,
    /// which is enforced at compile time via the `'graph` lifetime.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c)
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    #[cfg(cuda_11_only)]
    pub unsafe fn set_kernel_node_params(
        &mut self,
        node: &CudaGraphNode<'graph>,
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
    /// The node must come from the same graph this executable was instantiated from,
    /// which is enforced at compile time via the `'graph` lifetime.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c)
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    #[cfg(cuda_12_plus)]
    pub unsafe fn set_kernel_node_params(
        &mut self,
        node: &CudaGraphNode<'graph>,
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
    /// The node must come from the same graph this executable was instantiated from,
    /// which is enforced at compile time via the `'graph` lifetime.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c)
    ///
    /// # Safety
    ///
    /// - `args` must match the kernel signature exactly.
    /// - Pointers in `args` must remain valid until graph execution completes.
    pub unsafe fn set_kernel_node_args(
        &mut self,
        node: &CudaGraphNode<'graph>,
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
    /// The node must come from the same graph this executable was instantiated from,
    /// which is enforced at compile time via the `'graph` lifetime.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50a5c0a1a5a6b0c7b3e3d5a8a9c3b0d7)
    ///
    /// # Safety
    ///
    /// - `dst` and `src` must be valid device pointers.
    /// - The memory regions must remain valid until graph execution completes.
    pub unsafe fn set_memcpy_node_params(
        &mut self,
        node: &CudaGraphNode<'graph>,
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
    /// Returns an error if the graph definition belongs to a different context.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g27a7df53a4a5e4a9c3d4d3b5a8a9c3b0)
    pub fn update(&mut self, graph: &CudaGraphDef) -> Result<GraphUpdateResult, DriverError> {
        if !Arc::ptr_eq(&self.ctx, &graph.ctx) {
            return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT));
        }
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
    #[inline]
    pub fn cu_graph_exec(&self) -> sys::CUgraphExec {
        self.cu_graph_exec
    }
}
