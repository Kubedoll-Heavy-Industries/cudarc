# CUDA Graph Implementation Code Review

**Branch**: `cagyirey/dev`
**Primary files reviewed**: `src/driver/safe/graph.rs`, `src/driver/result.rs`, `src/driver/safe/mem_pool.rs`, `src/driver/safe/core.rs`
**Review date**: 2026-02-05

---

## Executive Summary

The CUDA graph implementation is functional but has significant issues that must be addressed before publishing. The most glaring problem is the repeated 15+ line `#[cfg(any(...))]` blocks scattered throughout the code. There are also lifetime soundness concerns, missing functionality, and inconsistent API patterns.

---

## 1. CRITICAL ISSUES (Must Fix Before Publish)

### 1.1 Feature Flag Duplication is Unmaintainable

**Severity**: Critical
**Location**: Throughout `graph.rs`, `result.rs`, `mod.rs`

The same 15-line `#[cfg(any(...))]` block appears **at least 12 times** across the graph implementation:

```rust
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
```

And the CUDA 12+ variant appears 6+ times:

```rust
#[cfg(any(
    feature = "cuda-12000",
    feature = "cuda-12010",
    // ... 11 more lines
))]
```

**Problems**:
1. When CUDA 12.10, 13.1, etc. releases, you need to update 20+ locations
2. Easy to miss one location, causing silent compilation failures
3. Inconsistent lists (some include 13.x, others don't)
4. The `edges()` function is missing CUDA 13.x features while other functions include them

**Solution**: See Section 4 for detailed fix.

---

### 1.2 Missing CUDA 13.x Support in Several Functions

**Severity**: Critical
**Location**: `graph.rs` lines 109-124, 471-486

The `capture_info()` and `edges()` methods are **missing CUDA 13.x features** in their cfg guards:

```rust
// capture_info - MISSING cuda-13000, cuda-13010
#[cfg(any(
    feature = "cuda-11040",
    // ...
    feature = "cuda-12090"  // <-- No 13.x!
))]
pub fn capture_info(&self) -> Result<result::stream::CaptureInfo, DriverError>
```

This will cause a **compilation error** for CUDA 13.x users who expect these APIs.

---

### 1.3 Lifetime Soundness Concern: `CudaGraphNode` Escapes Graph

**Severity**: Critical
**Location**: `graph.rs` lines 372-403

`CudaGraphNode<'graph>` has `Send + Sync` implemented unconditionally:

```rust
unsafe impl Send for CudaGraphNode<'_> {}
unsafe impl Sync for CudaGraphNode<'_> {}
```

**Problem**: The CUDA docs state graphs are NOT thread-safe:
> "Graph objects (cudaGraph_t, CUgraph) are not internally synchronized and must not be accessed concurrently from multiple threads."

A `CudaGraphNode` holds a raw `CUgraphNode` pointer. If the parent graph is modified/destroyed on another thread while you hold a node reference, you have UB.

The lifetime parameter `'graph` ties the node to the borrow of the graph, which helps, but the `Send + Sync` impls allow moving the node to another thread where the graph might not be accessible.

**Suggested fix**: Remove `Send + Sync` from `CudaGraphNode`, or document clearly that the node must not outlive any concurrent graph operations.

---

### 1.4 `CudaGraph` is `!Send` but Doesn't Implement `!Send`

**Severity**: High
**Location**: `graph.rs` lines 23-27

`CudaGraph` documents itself as NOT thread-safe but has no explicit `!Send`/`!Sync` marker:

```rust
/// This object is **NOT** thread safe.
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}
```

Since the struct contains `Arc<CudaStream>` which is `Send + Sync`, `CudaGraph` will auto-derive `Send + Sync` unless you prevent it.

**Fix**: Add:
```rust
impl !Send for CudaGraph {}
impl !Sync for CudaGraph {}
```
Or use `PhantomData<*const ()>` as a `!Send + !Sync` marker.

Same issue affects `CudaGraphDef` and `CudaGraphExec`.

---

### 1.5 Resource Leak on Error Path

**Severity**: High
**Location**: `graph.rs` `end_capture()` lines 55-70

```rust
pub fn end_capture(
    self: &Arc<Self>,
    flags: sys::CUgraphInstantiate_flags,
) -> Result<Option<CudaGraph>, DriverError> {
    self.ctx.bind_to_thread()?;
    let cu_graph = unsafe { result::stream::end_capture(self.cu_stream) }?;
    if cu_graph.is_null() {
        return Ok(None);
    }
    let cu_graph_exec = unsafe { result::graph::instantiate(cu_graph, flags) }?;  // <-- LEAK!
    Ok(Some(CudaGraph { ... }))
}
```

If `instantiate()` fails, `cu_graph` is leaked. The graph was successfully captured but never destroyed.

**Fix**:
```rust
let cu_graph_exec = match unsafe { result::graph::instantiate(cu_graph, flags) } {
    Ok(exec) => exec,
    Err(e) => {
        // Clean up the captured graph on instantiation failure
        let _ = unsafe { result::graph::destroy(cu_graph) };
        return Err(e);
    }
};
```

---

## 2. SUGGESTED IMPROVEMENTS (Should Fix)

### 2.1 Inconsistent API: `CudaGraph` vs `CudaGraphDef`/`CudaGraphExec` Pattern

The codebase has TWO different graph patterns:

1. **Legacy `CudaGraph`**: Combined graph + exec in one struct (lines 23-176)
2. **New `CudaGraphDef`/`CudaGraphExec`**: Separated definition and executable (lines 405-841)

This is confusing. The legacy pattern should be deprecated or removed.

**Recommendation**: Mark `CudaGraph` as `#[deprecated]` with a migration note:
```rust
#[deprecated(since = "0.20.0", note = "Use CudaGraphDef::instantiate() instead")]
pub struct CudaGraph { ... }
```

---

### 2.2 Missing `#[inline]` Hints

**Location**: Throughout `graph.rs`

Small accessor methods should be `#[inline]`:

```rust
// Should be:
#[inline]
pub fn cu_graph(&self) -> sys::CUgraph {
    self.cu_graph
}

#[inline]
pub fn context(&self) -> &Arc<CudaContext> {
    &self.ctx
}

#[inline]
pub fn is_success(&self) -> bool {
    self.result == sys::CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS
}
```

Other functions in `core.rs` use `#[inline]` (see `result::launch_kernel`), so this should be consistent.

---

### 2.3 `GraphUpdateResult::error_node` Should Return `CudaGraphNode`

**Location**: `graph.rs` lines 597-611

```rust
pub struct GraphUpdateResult {
    pub result: sys::CUgraphExecUpdateResult,
    /// Note: This is a raw handle and may be from a different graph.
    pub error_node: Option<sys::CUgraphNode>,  // <-- Raw handle!
}
```

The error node is a raw handle with no lifetime tracking. This is inconsistent with the rest of the API which wraps handles in typed structs.

**Problem**: Users can accidentally use this handle after the graph is destroyed.

**Suggested fix**: Either:
1. Return `Option<CudaGraphNode<'_>>` (needs lifetime from graph param)
2. Document the danger more prominently
3. Only expose the node type, not the raw handle

---

### 2.4 Unnecessary `MaybeUninit` Dance in `update_kernel_node_args`

**Location**: `graph.rs` lines 297-317

```rust
let mut current_params = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::uninit();
result::graph::kernel_node_get_params(node.cu_node, current_params.as_mut_ptr())?;
let mut current_params = current_params.assume_init();
```

This could use the more idiomatic pattern:
```rust
let mut current_params = std::mem::zeroed::<sys::CUDA_KERNEL_NODE_PARAMS>();
result::graph::kernel_node_get_params(node.cu_node, &mut current_params)?;
```

---

### 2.5 Missing Context Validation in Some Methods

**Location**: `graph.rs` `CudaGraphExec::launch()` validates context, but `CudaGraphExec::update()` doesn't

```rust
// launch() does this:
if self.ctx != &stream.ctx {
    return Err(DriverError(sys::cudaError_enum::CUDA_ERROR_INVALID_CONTEXT));
}

// update() should probably do similar:
pub fn update(&mut self, graph: &CudaGraphDef) -> Result<GraphUpdateResult, DriverError> {
    // Missing: if self.ctx != &graph.ctx { ... }
    self.ctx.bind_to_thread()?;
    ...
}
```

---

### 2.6 Documentation Links Are Broken/Vague

Several doc links point to non-existent anchors:

```rust
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50a5c0a1a5a6b0c7b3e3d5a8a9c3b0d7)
```

The anchor `g50a5c0a1a5a6b0c7b3e3d5a8a9c3b0d7` looks auto-generated and might not be stable. Use the function name anchor instead:
```rust
/// See [cuGraphExecMemcpyNodeSetParams](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga0e02c6f4da3c19ff4c20c99ff5f3b4a)
```

---

## 3. NICE-TO-HAVES (Future Work)

### 3.1 Missing Graph Node Addition APIs

Users can only create graphs via stream capture. Programmatic graph building is missing:

- `add_kernel_node()`
- `add_memcpy_node()`
- `add_memset_node()`
- `add_host_node()`
- `add_event_node()` (CUDA 12+)
- `add_empty_node()`
- `add_child_graph_node()`
- `add_dependencies()`/`remove_dependencies()`

These are essential for advanced graph workflows.

### 3.2 Missing Graph Debug APIs

- `cuGraphDebugDotPrint` - Export graph to DOT format for visualization
- Node introspection (get dependencies, get kernel params, etc.)

### 3.3 Missing `Debug` Implementations

`CudaGraph`, `CudaGraphDef`, `CudaGraphExec` don't implement `Debug`:

```rust
#[derive(Debug)]  // Missing
pub struct CudaGraph { ... }
```

This makes debugging difficult.

### 3.4 Missing Tests

The graph module has **zero tests**. At minimum:
- Test capture/instantiate/launch cycle
- Test graph cloning
- Test parameter updates
- Test error handling (invalid node types, etc.)
- Test the lifetime guarantees compile-fail

### 3.5 Missing `Clone` for `CudaGraphDef`

`CudaGraphDef::try_clone()` exists but `Clone` trait isn't implemented. Consider:
```rust
impl Clone for CudaGraphDef {
    fn clone(&self) -> Self {
        self.try_clone().expect("Graph clone failed")
    }
}
```
Or document why `Clone` isn't implemented (to force error handling).

---

## 4. SOLVING THE FEATURE FLAG PROBLEM

### Option A: Macro-based (Recommended)

Create a macro in `src/driver/mod.rs` or a dedicated `src/macros.rs`:

```rust
/// Expands to `#[cfg(any(feature = "cuda-11040", ...))]` for all supported versions.
macro_rules! cfg_cuda_11_4_plus {
    ($($item:item)*) => {
        $(
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
                feature = "cuda-12090",
                feature = "cuda-13000",
                feature = "cuda-13010",
            ))]
            $item
        )*
    };
}

/// For CUDA 12.0+ only features
macro_rules! cfg_cuda_12_plus {
    ($($item:item)*) => {
        $(
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
                feature = "cuda-13010",
            ))]
            $item
        )*
    };
}

/// For CUDA 11.x only (deprecated APIs)
macro_rules! cfg_cuda_11_only {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080",
            ))]
            $item
        )*
    };
}
```

**Usage**:
```rust
cfg_cuda_11_4_plus! {
    pub fn capture_info(&self) -> Result<CaptureInfo, DriverError> {
        // ...
    }
}

cfg_cuda_11_only! {
    pub unsafe fn update_kernel_node_params(...) {
        // CUDA 11.x version
    }
}

cfg_cuda_12_plus! {
    pub unsafe fn update_kernel_node_params(...) {
        // CUDA 12+ version with extra fields
    }
}
```

### Option B: build.rs cfg flags (More Elegant)

Modify `build.rs` to emit semantic cfg flags:

```rust
// In build.rs
fn main() {
    // ... existing code ...

    // Emit semantic version flags
    if major >= 11 && minor >= 4 {
        println!("cargo:rustc-cfg=cuda_11_4_plus");
    }
    if major >= 12 {
        println!("cargo:rustc-cfg=cuda_12_plus");
    }
    if major >= 13 {
        println!("cargo:rustc-cfg=cuda_13_plus");
    }
    if major == 11 {
        println!("cargo:rustc-cfg=cuda_11_only");
    }
}
```

**Usage**:
```rust
#[cfg(cuda_11_4_plus)]
pub fn capture_info(&self) -> Result<CaptureInfo, DriverError> { ... }

#[cfg(cuda_11_only)]
pub unsafe fn update_kernel_node_params(...) { /* v1 */ }

#[cfg(cuda_12_plus)]
pub unsafe fn update_kernel_node_params(...) { /* v2 */ }
```

This is cleaner but requires users to understand the build.rs magic.

### Option C: Version Comparison at Runtime

For APIs that exist in all versions but have different behavior:
```rust
const CUDA_MAJOR: usize = env!("CUDA_MAJOR_VERSION").parse().unwrap();

pub fn some_function() {
    if CUDA_MAJOR >= 12 {
        // CUDA 12+ path
    } else {
        // CUDA 11 path
    }
}
```

**Not recommended** - adds runtime overhead and doesn't work for signature differences.

---

## 5. SUMMARY OF REQUIRED CHANGES

### Before Publishing:
1. [x] Fix missing CUDA 13.x in `capture_info()` and `edges()` cfg guards - **DONE**: Now uses `#[cfg(cuda_11_4_plus)]`
2. [x] Add `!Send + !Sync` markers to `CudaGraph`, `CudaGraphDef`, `CudaGraphExec` - **DONE**: Added `PhantomData<*const ()>` field
3. [x] Remove `Send + Sync` from `CudaGraphNode` or document the danger - **DONE**: Removed unsafe impls, added comment
4. [x] Fix resource leak in `end_capture()` when `instantiate()` fails - **DONE**: Now cleans up `cu_graph` on error
5. [x] Implement one of the feature flag solutions (Option A or B) - **DONE**: Implemented Option B (build.rs semantic cfg flags)

### Should Fix:
6. [ ] Deprecate `CudaGraph` in favor of `CudaGraphDef`/`CudaGraphExec`
7. [ ] Add `#[inline]` to accessor methods
8. [ ] Add context validation to `CudaGraphExec::update()`
9. [ ] Fix broken documentation links
10. [ ] Add `Debug` impl to graph types

### Future Work:
11. [ ] Add programmatic graph building APIs
12. [ ] Add graph debug/export APIs
13. [ ] Add comprehensive tests
14. [ ] Consider `Clone` trait for `CudaGraphDef`

---

## Appendix: Files Changed Summary

| File | Lines Changed | Issues |
|------|---------------|--------|
| `src/driver/safe/graph.rs` | 842 | Lifetime holes, feature flag duplication, resource leak |
| `src/driver/result.rs` | ~350 (graph module) | Feature flag duplication, missing 13.x |
| `src/driver/safe/mod.rs` | 42 | Feature flag duplication |
| `build.rs` | Unmodified | Should emit semantic cfg flags |
