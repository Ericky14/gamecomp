//! GPU utilities shared across backends.
//!
//! Contains Vulkan-based helpers that are not backends themselves but provide
//! GPU-accelerated operations (e.g., DMA-BUF blitting) for backend
//! implementations that need them.

pub mod vulkan_blitter;
