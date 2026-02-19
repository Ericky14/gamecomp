//! Post-processing pipeline abstraction.
//!
//! Post-processors are optional filters applied after composition but before
//! presentation. They are chained together and each operates on the output
//! of the previous stage.
//!
//! Feature-gated implementations:
//! - FSR 1.0 (AMD FidelityFX Super Resolution) — `fsr` feature
//! - NIS (NVIDIA Image Scaling) — `nis` feature
//! - CAS (Contrast Adaptive Sharpening) — `cas` feature
//!
//! Post-processors are hot-swappable at runtime via configuration changes.

use anyhow::Result;

use super::ExportedFrame;

/// A post-processing filter in the render pipeline.
///
/// Post-processors receive a composited frame and produce a modified frame.
/// They may change resolution (upscaling) or only modify pixel values
/// (sharpening, color grading).
pub trait PostProcessor: Send {
    /// Human-readable name (e.g., "FSR 1.0", "NIS").
    fn name(&self) -> &str;

    /// Whether this post-processor changes the output resolution.
    fn changes_resolution(&self) -> bool;

    /// Process a frame. The input frame is consumed and a new frame is returned.
    ///
    /// The implementation may modify the frame in-place (if the backing storage
    /// is owned) or allocate a new output buffer.
    fn process(&mut self, frame: &ExportedFrame) -> Result<ExportedFrame>;

    /// Update parameters (e.g., sharpness, upscale ratio). Called when config changes.
    fn configure(&mut self, params: &PostProcessParams) -> Result<()>;
}

/// Configuration parameters for post-processing.
#[derive(Debug, Clone, Copy)]
pub struct PostProcessParams {
    /// Sharpness intensity (0.0 = off, 1.0 = maximum). Used by FSR, NIS, CAS.
    pub sharpness: f32,
    /// Target output resolution for upscaling. `None` = same as input.
    pub target_resolution: Option<(u32, u32)>,
}

impl Default for PostProcessParams {
    fn default() -> Self {
        Self {
            sharpness: 0.5,
            target_resolution: None,
        }
    }
}

/// A chain of post-processors applied in sequence.
///
/// The chain is evaluated lazily — if empty, the composited frame passes
/// through unchanged with zero overhead.
pub struct PostProcessChain {
    /// Ordered list of active post-processors.
    stages: Vec<Box<dyn PostProcessor>>,
}

impl PostProcessChain {
    /// Create an empty chain (pass-through).
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a post-processor to the end of the chain.
    pub fn push(&mut self, stage: Box<dyn PostProcessor>) {
        self.stages.push(stage);
    }

    /// Remove all post-processors.
    pub fn clear(&mut self) {
        self.stages.clear();
    }

    /// Whether the chain is empty (pure pass-through).
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Number of active stages.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.stages.len()
    }
}

impl Default for PostProcessChain {
    fn default() -> Self {
        Self::new()
    }
}
