//! Per-run configuration.

use std::num::NonZeroU32;
use std::sync::Arc;

/// The default recursion limit: 10,007 supersteps, as in current LangGraph.
pub const DEFAULT_RECURSION_LIMIT: NonZeroU32 = NonZeroU32::MIN.saturating_add(10_006);

/// The configuration of one run: its typed context and its recursion limit.
#[derive(Debug)]
pub struct RunConfig<C = ()> {
    /// The run's context, shared by every task.
    pub(crate) context: Arc<C>,
    /// The number of supersteps the run may take.
    pub(crate) recursion_limit: NonZeroU32,
}

impl<C> RunConfig<C> {
    /// A configuration carrying `context`, with the default recursion limit.
    #[must_use]
    pub fn new(context: C) -> Self {
        Self {
            context: Arc::new(context),
            recursion_limit: DEFAULT_RECURSION_LIMIT,
        }
    }

    /// Sets the number of supersteps the run may take.
    ///
    /// A run fails with [`RunError::RecursionLimit`](crate::RunError::RecursionLimit)
    /// if a task is planned for superstep number `limit`, counting from 0; a
    /// graph that finishes in exactly `limit` supersteps succeeds. The limit is
    /// non-zero, so a run that cannot take a single step is unrepresentable.
    #[must_use]
    pub const fn with_recursion_limit(mut self, limit: NonZeroU32) -> Self {
        self.recursion_limit = limit;
        self
    }
}

impl<C> Clone for RunConfig<C> {
    /// Another configuration sharing the same context, without requiring
    /// `C: Clone`.
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            recursion_limit: self.recursion_limit,
        }
    }
}

impl<C: Default> Default for RunConfig<C> {
    /// A configuration with the default context and recursion limit.
    fn default() -> Self {
        Self::new(C::default())
    }
}

#[cfg(test)]
mod tests {
    //! Tests for run configurations.

    use std::any::TypeId;

    use super::*;

    // --- RunConfig -----------------------------------------------------------

    /// A run without a context never has to name `C`: it defaults to `()`.
    #[test]
    fn the_context_defaults_to_unit() {
        assert_eq!(TypeId::of::<RunConfig>(), TypeId::of::<RunConfig<()>>());
    }
}
