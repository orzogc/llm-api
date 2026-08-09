//! Provider format implementations.
//!
//! Each module owns complete serde types for its wire format plus IR
//! conversion functions (typed layer), and an [`crate::ApiFormat`]
//! implementation (dynamic layer).

pub mod anthropic_messages;
pub mod google_generate_content;
pub mod openai_chat_completions;
pub mod openai_responses;
