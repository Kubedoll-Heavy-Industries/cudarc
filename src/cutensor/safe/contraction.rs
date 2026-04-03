use super::{result, sys};

/// Contraction operation descriptor.
pub struct ContractionDescriptor {
    pub(crate) handle: sys::cutensorHandle_t,
    pub(crate) desc: sys::cutensorOperationDescriptor_t,
}

impl Drop for ContractionDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { result::destroy_operation_descriptor(self.desc) }.ok();
        }
    }
}

/// Plan preference for cuTENSOR operations.
pub struct PlanPreference {
    pub(crate) handle: sys::cutensorHandle_t,
    pub(crate) pref: sys::cutensorPlanPreference_t,
}

impl Drop for PlanPreference {
    fn drop(&mut self) {
        if !self.pref.is_null() {
            unsafe { result::destroy_plan_preference(self.pref) }.ok();
        }
    }
}

/// Execution plan for cuTENSOR operations.
pub struct Plan {
    pub(crate) handle: sys::cutensorHandle_t,
    pub(crate) plan: sys::cutensorPlan_t,
}

impl Drop for Plan {
    fn drop(&mut self) {
        if !self.plan.is_null() {
            unsafe { result::destroy_plan(self.plan) }.ok();
        }
    }
}
