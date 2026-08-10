//! Request hooks (§ 5) — closures live in configuration, not in IR nodes.

use std::sync::Arc;

use serde_json::Value;

use crate::error::HookError;
use crate::ir::Role;

/// Type of the per-message hook: `(serialized message index, role, &mut
/// message JSON)`.
pub type MessageHook = dyn Fn(usize, &Role, &mut Value) -> Result<(), HookError> + Send + Sync;

/// Type of the whole-request hook.
pub type RequestHook = dyn Fn(&mut Value) -> Result<(), HookError> + Send + Sync;

/// Hooks that run after IR→JSON serialization (including the `extra` merge
/// and the strict gate) and before sending. A hook returning `Err` aborts
/// the call with `Error::Hook`. Post-hook JSON is not re-validated —
/// warnings describe the conversion stage only; hook consequences are the
/// user's (§ 6).
#[non_exhaustive]
#[derive(Clone, Default)]
pub struct RequestHooks {
    /// Runs once per serialized message. The index and role refer to the
    /// serialized **target-format** message sequence (after the splits,
    /// merges and downgrades of § 7); top-level system channels (Anthropic
    /// `system`, Responses `instructions`, Google `systemInstruction`) are
    /// not visited — use `on_request` for those. The (index, role) pairs are
    /// a snapshot taken at serialization time: a message whose pointer — or
    /// wire `role` key — a request- or message-level `extra` merge rewrote
    /// is skipped (the snapshot would no longer match the wire), and
    /// messages `extra` injected are not visited — rewriting messages that
    /// way is `on_request` territory.
    pub on_message: Option<Arc<MessageHook>>,
    /// Runs on the final request JSON before sending.
    pub on_request: Option<Arc<RequestHook>>,
}

impl RequestHooks {
    /// No hooks.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the per-message hook.
    #[must_use]
    pub fn with_on_message(
        mut self,
        hook: impl Fn(usize, &Role, &mut Value) -> Result<(), HookError> + Send + Sync + 'static,
    ) -> Self {
        self.on_message = Some(Arc::new(hook));
        self
    }

    /// Sets the whole-request hook.
    #[must_use]
    pub fn with_on_request(
        mut self,
        hook: impl Fn(&mut Value) -> Result<(), HookError> + Send + Sync + 'static,
    ) -> Self {
        self.on_request = Some(Arc::new(hook));
        self
    }

    /// `true` when no hook is set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.on_message.is_none() && self.on_request.is_none()
    }
}

impl std::fmt::Debug for RequestHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestHooks")
            .field("on_message", &self.on_message.as_ref().map(|_| "…"))
            .field("on_request", &self.on_request.as_ref().map(|_| "…"))
            .finish()
    }
}
