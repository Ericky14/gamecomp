//! Color management pipeline.
//!
//! Defines the [`ColorPipeline`] trait for color space conversion, HDR
//! tonemapping, and LUT (Look-Up Table) application. The pipeline runs
//! as part of the composition stage — the renderer calls into it when
//! blending layers with different color spaces.
//!
//! Design: Color management is decomposed into pluggable transforms that
//! can be composed into a pipeline. Each transform handles one conversion
//! step (e.g., PQ→sRGB, gamut compression, LUT application).
//!
//! Feature-gated behind the `hdr` feature flag. When disabled, the
//! pipeline is a no-op and compiles to zero code.

use anyhow::Result;

/// Supported color transfer functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferFunction {
    /// sRGB gamma (~2.2 with linear toe).
    #[default]
    Srgb,
    /// Linear (gamma 1.0).
    Linear,
    /// SMPTE ST.2084 Perceptual Quantizer (HDR).
    Pq,
    /// Hybrid Log-Gamma (HLG, broadcast HDR).
    Hlg,
    /// Power-law gamma (custom exponent in metadata).
    Gamma,
}

/// Supported color gamuts (primaries).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorGamut {
    /// sRGB / BT.709 (standard desktop).
    #[default]
    Srgb,
    /// DCI-P3 (wide gamut, used in HDR displays).
    DciP3,
    /// BT.2020 (ultra-wide gamut, 10-bit HDR content).
    Bt2020,
}

/// Describes the color properties of a surface or output.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColorDescription {
    /// Transfer function (EOTF).
    pub transfer: TransferFunction,
    /// Color gamut / primaries.
    pub gamut: ColorGamut,
    /// Maximum luminance in nits (for HDR content). 0 = unknown/SDR.
    pub max_luminance: f32,
    /// Minimum luminance in nits. 0 = unknown/SDR.
    pub min_luminance: f32,
    /// Maximum content light level (MaxCLL) in nits. 0 = unknown.
    pub max_cll: f32,
    /// Maximum frame-average light level (MaxFALL) in nits. 0 = unknown.
    pub max_fall: f32,
}

impl ColorDescription {
    /// Whether this describes HDR content.
    #[inline(always)]
    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer, TransferFunction::Pq | TransferFunction::Hlg)
    }

    /// Standard SDR sRGB.
    pub const SDR: Self = Self {
        transfer: TransferFunction::Srgb,
        gamut: ColorGamut::Srgb,
        max_luminance: 0.0,
        min_luminance: 0.0,
        max_cll: 0.0,
        max_fall: 0.0,
    };
}

/// A color management pipeline that maps source content to output display color.
///
/// Implementations handle the GPU shader code or LUT data needed for color
/// conversion. The pipeline is called during composition to transform each
/// layer's pixels from its source color space to the output color space.
pub trait ColorPipeline: Send {
    /// Configure the pipeline for a given source → output mapping.
    ///
    /// Called when the source content color properties change (e.g., an HDR
    /// game starts) or the output display capabilities change.
    fn configure(&mut self, source: &ColorDescription, output: &ColorDescription) -> Result<()>;

    /// Whether the pipeline is a no-op (source matches output).
    fn is_identity(&self) -> bool;

    /// Load a 3D LUT (Look-Up Table) for color correction.
    ///
    /// The LUT data is a flattened 3D cube in row-major order.
    /// `size` is the cube dimension (e.g., 33 for a 33×33×33 LUT).
    fn load_lut(&mut self, data: &[f32], size: u32) -> Result<()>;

    /// Clear any loaded LUT.
    fn clear_lut(&mut self);
}

/// No-op color pipeline for SDR-only mode (compiles to zero code).
///
/// Used when the `hdr` feature is disabled or when source and output
/// are both standard sRGB.
pub struct IdentityColorPipeline;

impl ColorPipeline for IdentityColorPipeline {
    fn configure(&mut self, _source: &ColorDescription, _output: &ColorDescription) -> Result<()> {
        Ok(())
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        true
    }

    fn load_lut(&mut self, _data: &[f32], _size: u32) -> Result<()> {
        Ok(())
    }

    fn clear_lut(&mut self) {}
}
