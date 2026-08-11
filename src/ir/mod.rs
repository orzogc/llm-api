//! The unified intermediate representation (IR).
//!
//! All IR data types are serializable — persisting agent history as IR JSON
//! is a supported use case, and the IR JSON representation is covered by
//! semver (§ 3 of the design document). The two process-state types exported
//! here ([`Accumulator`], [`MergeLog`]) are not persistable IR data and
//! carry no serde support.

mod block;
mod events;
mod extra;
mod message;
mod output;
mod reasoning;
mod request;
mod response;
mod role;
mod tools;

pub use block::{CacheHint, ContentBlock, ImageSource, ToolOutputBlock};
pub use events::{Accumulator, BlockDelta, StreamEvent, StreamItem};
pub use extra::{Extra, MergeLog, escape_pointer_token, merge_patch};
pub use message::{Message, RoundTripMeta};
pub use output::OutputFormat;
pub use reasoning::{Effort, Reasoning};
pub use request::Request;
pub use response::{
    REFUSAL_MARKER, Response, StopReason, Usage, is_refusal_block, normalize_stop_reason,
};
pub use role::Role;
pub use tools::{FunctionTool, Tool, ToolChoice};
