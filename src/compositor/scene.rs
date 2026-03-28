//! Scene description types for the composition pipeline.
//!
//! These types describe *what* to render each frame. The main thread builds a
//! [`FrameInfo`] and sends it to the render thread via channel. All types are
//! `Copy` and stack-allocated — zero heap allocation per frame.
//!
//! Design: Fixed-size arrays with a layer count. Max 4 layers covers all
//! realistic scenarios (app + overlay + cursor + video underlay). The struct
//! is small enough to pass by value through a channel.

use drm_fourcc::DrmFourcc;

/// Maximum number of composition layers per frame.
pub const MAX_LAYERS: usize = 4;

/// Layer indices by convention.
pub const LAYER_VIDEO_UNDERLAY: usize = 0;
pub const LAYER_APP: usize = 1;
pub const LAYER_OVERLAY: usize = 2;
pub const LAYER_CURSOR: usize = 3;

/// Filter mode for texture sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum FilterMode {
    /// Nearest-neighbor sampling (no interpolation).
    #[default]
    Nearest = 0,
    /// Bilinear filtering.
    Linear = 1,
}

/// Color space of a layer's content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum ColorSpace {
    /// Standard sRGB with gamma encoding.
    #[default]
    Srgb = 0,
    /// Linear sRGB (no gamma).
    LinearSrgb = 1,
    /// HDR PQ (Perceptual Quantizer, ST.2084).
    Pq = 2,
    /// scRGB (linear, extended range).
    ScRgb = 3,
}

/// Alpha blending mode for a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum BlendMode {
    /// No blending — layer is opaque.
    #[default]
    Opaque = 0,
    /// Standard alpha blending (premultiplied alpha).
    AlphaPreMultiplied = 1,
    /// Straight alpha blending.
    AlphaStraight = 2,
}

/// A rectangle in pixels.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// A single composition layer.
///
/// Describes one texture to be composited into the output.
/// All coordinates are in output pixel space.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Layer {
    /// Index into the texture array (set by the render thread).
    pub texture_index: u32,
    /// Whether this layer is active (should be composited).
    pub active: bool,
    /// Source rectangle within the texture (for cropping/viewport).
    pub src: Rect,
    /// Destination rectangle on the output.
    pub dst: Rect,
    /// Opacity (0.0 = fully transparent, 1.0 = fully opaque).
    pub opacity: f32,
    /// Texture sampling filter.
    pub filter: FilterMode,
    /// Color space of the layer content.
    pub color_space: ColorSpace,
    /// Alpha blending mode.
    pub blend: BlendMode,
    /// DRM format of the source buffer.
    pub format: u32,
}

/// Push constant data sent to the compute shader.
///
/// Must match the layout in `composite.comp`. Kept minimal
/// to fit within the guaranteed 128-byte push constant limit.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct CompositeParams {
    /// Output dimensions.
    pub output_width: u32,
    pub output_height: u32,
    /// Number of active layers.
    pub layer_count: u32,
    /// Padding for alignment.
    pub _pad: u32,
    // Per-layer parameters (position, opacity, flags).
    // Each layer: src_x, src_y, src_w, src_h, dst_x, dst_y, dst_w, dst_h, opacity, flags, pad, pad
    // = 12 u32s per layer × 4 layers = 192 bytes — exceeds 128-byte limit.
    // So we use a UBO for layers instead. Push constants hold only globals.
}

/// Composition mode for the current frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositionMode {
    /// Direct scanout — client buffer goes straight to hardware plane.
    /// No GPU work needed.
    DirectScanout,
    /// Vulkan compute composition — one or more layers composited via shader.
    Composite,
    /// Upscale then composite (FSR or NIS).
    Upscale,
    /// Skip this frame (nothing changed).
    Skip,
}

/// Complete scene description for one frame.
///
/// Built by the main thread, sent to the render thread by value.
/// Fixed-size, `Copy`, zero allocation.
#[derive(Debug, Clone, Copy)]
pub struct FrameInfo {
    /// Composition layers. Only `layer_count` entries are valid.
    pub layers: [Layer; MAX_LAYERS],
    /// Number of active layers (0..=MAX_LAYERS).
    pub layer_count: u32,
    /// Output dimensions.
    pub output_width: u32,
    pub output_height: u32,
    /// Desired composition mode.
    pub mode: CompositionMode,
    /// Frame sequence number (monotonically increasing).
    pub seq: u64,
    /// Target presentation timestamp (nanoseconds, CLOCK_MONOTONIC).
    pub target_present_ns: u64,
    /// Whether VRR is active for this frame.
    pub vrr_active: bool,
}

impl Default for FrameInfo {
    #[inline(always)]
    fn default() -> Self {
        Self {
            layers: [Layer::default(); MAX_LAYERS],
            layer_count: 0,
            output_width: 0,
            output_height: 0,
            mode: CompositionMode::Skip,
            seq: 0,
            target_present_ns: 0,
            vrr_active: false,
        }
    }
}

impl FrameInfo {
    /// Create a frame with a single fullscreen layer (direct scanout candidate).
    #[inline(always)]
    pub fn single_layer(width: u32, height: u32, texture_index: u32, format: DrmFourcc) -> Self {
        let mut layers = [Layer::default(); MAX_LAYERS];
        layers[LAYER_APP] = Layer {
            texture_index,
            active: true,
            src: Rect {
                x: 0,
                y: 0,
                width,
                height,
            },
            dst: Rect {
                x: 0,
                y: 0,
                width,
                height,
            },
            opacity: 1.0,
            filter: FilterMode::Nearest,
            color_space: ColorSpace::Srgb,
            blend: BlendMode::Opaque,
            format: format as u32,
        };
        Self {
            layers,
            layer_count: (LAYER_APP + 1) as u32,
            output_width: width,
            output_height: height,
            mode: CompositionMode::DirectScanout,
            seq: 0,
            target_present_ns: 0,
            vrr_active: false,
        }
    }

    /// Get the active layers as a slice.
    #[inline(always)]
    pub fn active_layers(&self) -> &[Layer] {
        // SAFETY: layer_count is always <= MAX_LAYERS, enforced by construction.
        unsafe {
            core::hint::assert_unchecked(self.layer_count as usize <= MAX_LAYERS);
        }
        &self.layers[..self.layer_count as usize]
    }

    /// Returns true if this frame can potentially use direct scanout.
    ///
    /// Direct scanout is possible when there is exactly one active layer that
    /// covers the full output, is fully opaque, and uses the app slot.
    #[inline(always)]
    pub fn is_direct_scanout_candidate(&self) -> bool {
        // Count active layers — only one must be active and it must be the app layer.
        let active_count = self.layers.iter().filter(|l| l.active).count();
        active_count == 1
            && self.layers[LAYER_APP].active
            && self.layers[LAYER_APP].blend == BlendMode::Opaque
            && self.layers[LAYER_APP].opacity >= 1.0
            && self.layers[LAYER_APP].src.width == self.output_width
            && self.layers[LAYER_APP].src.height == self.output_height
            && self.layers[LAYER_APP].dst.width == self.output_width
            && self.layers[LAYER_APP].dst.height == self.output_height
    }
}

#[cfg(test)]
#[path = "scene_tests.rs"]
mod tests;
