//! DRM format modifier queries and filtering.
//!
//! Enumerates Vulkan-supported DRM modifiers for B8G8R8A8_UNORM and filters
//! them by external memory capability (exportable for self-allocated outputs,
//! importable for GBM-allocated outputs).

use super::*;

impl VulkanBlitter {
    /// Two-pass query of DRM format modifier properties for `VK_FORMAT_XRGB`.
    ///
    /// First call queries the count, second call fills the properties array.
    /// Used by all modifier query / filter functions.
    fn query_drm_modifier_props(&self) -> anyhow::Result<Vec<vk::DrmFormatModifierPropertiesEXT>> {
        let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut format_props2 = vk::FormatProperties2::default().push_next(&mut modifier_list);
        // SAFETY: physical_device is valid; first call queries count only.
        unsafe {
            self.instance.get_physical_device_format_properties2(
                self.physical_device,
                VK_FORMAT_XRGB,
                &mut format_props2,
            );
        }

        let count = modifier_list.drm_format_modifier_count as usize;
        let mut modifier_props = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
        let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut modifier_props);
        let mut format_props2 = vk::FormatProperties2::default().push_next(&mut modifier_list);
        // SAFETY: physical_device is valid; modifier_props has `count` elements.
        unsafe {
            self.instance.get_physical_device_format_properties2(
                self.physical_device,
                VK_FORMAT_XRGB,
                &mut format_props2,
            );
        }

        Ok(modifier_props)
    }

    /// Filter modifiers by external memory capability (EXPORTABLE or IMPORTABLE).
    ///
    /// For each candidate modifier that passes `required_features` pre-filtering,
    /// queries `vkGetPhysicalDeviceImageFormatProperties2` with
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT` to verify the modifier
    /// supports the requested `external_feature`. Returns modifiers intersected
    /// with `scanout_modifiers` when non-empty.
    fn filter_modifiers_by_external_capability(
        &self,
        scanout_modifiers: &[u64],
        width: u32,
        height: u32,
        required_features: vk::FormatFeatureFlags,
        external_feature: vk::ExternalMemoryFeatureFlags,
        label: &str,
    ) -> anyhow::Result<Vec<u64>> {
        let modifier_props = self.query_drm_modifier_props()?;

        // Pre-filter: single-plane, non-INVALID, has required tiling features.
        let candidates: Vec<u64> = modifier_props
            .iter()
            .filter(|mp| {
                mp.drm_format_modifier != DRM_FORMAT_MOD_INVALID
                    && mp.drm_format_modifier_plane_count == 1
                    && mp
                        .drm_format_modifier_tiling_features
                        .contains(required_features)
            })
            .map(|mp| mp.drm_format_modifier)
            .collect();

        // For each candidate, verify DMA-BUF export/import capability via
        // vkGetPhysicalDeviceImageFormatProperties2.
        let mut compatible: Vec<u64> = Vec::new();
        for &modifier in &candidates {
            let mut drm_mod_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
                .drm_format_modifier(modifier)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let mut external_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let image_format_info = vk::PhysicalDeviceImageFormatInfo2::default()
                .format(VK_FORMAT_XRGB)
                .ty(vk::ImageType::TYPE_2D)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(OUTPUT_IMAGE_USAGE)
                .push_next(&mut drm_mod_info)
                .push_next(&mut external_info);

            let mut external_props = vk::ExternalImageFormatProperties::default();
            let mut image_format_props =
                vk::ImageFormatProperties2::default().push_next(&mut external_props);

            // SAFETY: physical_device is valid; image_format_info is fully
            // initialized with modifier and external handle type chains.
            let result = unsafe {
                self.instance.get_physical_device_image_format_properties2(
                    self.physical_device,
                    &image_format_info,
                    &mut image_format_props,
                )
            };

            if result.is_err() {
                debug!(
                    modifier = format!("0x{:016x}", modifier),
                    "blitter: modifier not supported for {label} usage, skipping"
                );
                continue;
            }

            let max = image_format_props.image_format_properties.max_extent;
            let ext_features = external_props
                .external_memory_properties
                .external_memory_features;

            if !ext_features.contains(external_feature) {
                debug!(
                    modifier = format!("0x{:016x}", modifier),
                    ?ext_features,
                    "blitter: modifier not {label}, skipping"
                );
                continue;
            }

            if max.width < width || max.height < height {
                debug!(
                    modifier = format!("0x{:016x}", modifier),
                    max_w = max.width,
                    max_h = max.height,
                    "blitter: {label} modifier max extent too small, skipping"
                );
                continue;
            }

            compatible.push(modifier);
        }

        info!(
            candidates = candidates.len(),
            compatible = compatible.len(),
            "blitter: verified per-modifier DMA-BUF {label} capability"
        );

        if !scanout_modifiers.is_empty() {
            let intersection: Vec<u64> = scanout_modifiers
                .iter()
                .copied()
                .filter(|m| compatible.contains(m))
                .collect();

            if !intersection.is_empty() {
                info!(
                    scanout = scanout_modifiers.len(),
                    compatible = compatible.len(),
                    intersection = intersection.len(),
                    "blitter: intersected DRM plane + Vulkan {label} modifiers"
                );
                return Ok(intersection);
            }

            warn!(
                scanout = scanout_modifiers.len(),
                compatible = compatible.len(),
                "blitter: no intersection between DRM plane and Vulkan {label} modifiers"
            );
        }

        Ok(compatible)
    }

    /// Query all valid DRM modifiers for B8G8R8A8_UNORM that support
    /// TRANSFER_SRC and SAMPLED_IMAGE.
    ///
    /// These are used when importing client DMA-BUFs — we provide the full list
    /// to `ImageDrmFormatModifierListCreateInfoEXT` and let the driver determine
    /// which modifier matches the imported buffer's actual tiling layout.
    /// SAMPLED_IMAGE is required for compute shader composition.
    pub(super) fn query_all_valid_modifiers(&self) -> anyhow::Result<Vec<u64>> {
        let modifier_props = self.query_drm_modifier_props()?;

        let required = IMPORT_MODIFIER_FEATURES;

        let modifiers: Vec<u64> = modifier_props
            .iter()
            .filter(|mp| {
                mp.drm_format_modifier != DRM_FORMAT_MOD_INVALID
                    && mp.drm_format_modifier_plane_count == 1
                    && mp.drm_format_modifier_tiling_features.contains(required)
            })
            .map(|mp| mp.drm_format_modifier)
            .collect();

        info!(
            count = modifiers.len(),
            "blitter: import modifiers (TRANSFER_SRC+SAMPLED_IMAGE, single-plane, non-INVALID)"
        );
        for m in &modifiers {
            debug!(
                modifier = format!("0x{:016x}", m),
                "blitter: import modifier"
            );
        }

        if modifiers.is_empty() {
            bail!("no valid import modifiers found");
        }

        Ok(modifiers)
    }

    /// Query the best explicit modifier for XRGB8888 output images.
    fn query_output_modifier(&self) -> anyhow::Result<u64> {
        let modifier_props = self.query_drm_modifier_props()?;

        if modifier_props.is_empty() {
            bail!("no DRM format modifiers available for B8G8R8A8_UNORM");
        }

        info!(
            count = modifier_props.len(),
            "blitter: available DRM modifiers for B8G8R8A8_UNORM"
        );

        // Pick the best modifier that supports:
        // - TRANSFER_DST (we blit into it)
        // - Single plane (simpler)
        // - NOT DRM_FORMAT_MOD_INVALID (need a concrete modifier for export)
        // - NOT LINEAR (poor performance)
        // Prefer vendor-specific tiled modifiers.
        let required_features = IMPORTABLE_MODIFIER_FEATURES;

        let mut best: Option<(u64, u32)> = None; // (modifier, plane_count)
        for mp in &modifier_props {
            let modifier = mp.drm_format_modifier;
            let planes = mp.drm_format_modifier_plane_count;
            let features = mp.drm_format_modifier_tiling_features;

            debug!(
                modifier = format!("0x{:016x}", modifier),
                planes,
                features = format!("0x{:x}", features.as_raw()),
                "blitter: modifier candidate"
            );

            if modifier == DRM_FORMAT_MOD_INVALID {
                continue;
            }
            if planes != 1 {
                continue; // Stick to single-plane for simplicity.
            }
            if !features.contains(required_features) {
                continue;
            }

            // Prefer non-LINEAR (tiled) over LINEAR.
            if best.is_none_or(|(existing, _)| {
                existing == DRM_FORMAT_MOD_LINEAR && modifier != DRM_FORMAT_MOD_LINEAR
            }) {
                best = Some((modifier, planes));
            }
        }

        let (modifier, _planes) = best.context(
            "no suitable DRM modifier found (need TRANSFER_DST, single-plane, non-INVALID)",
        )?;

        info!(
            modifier = format!("0x{:016x}", modifier),
            "blitter: selected output modifier"
        );

        Ok(modifier)
    }

    /// Compute the final list of modifiers for output image allocation.
    ///
    /// Verifies each modifier supports DMA-BUF export via
    /// `vkGetPhysicalDeviceImageFormatProperties2`. Falls back to
    /// `query_output_modifier()` if no exportable modifiers are found.
    pub(super) fn compute_output_modifiers(
        &self,
        scanout_modifiers: &[u64],
        width: u32,
        height: u32,
    ) -> anyhow::Result<Vec<u64>> {
        let result = self.filter_modifiers_by_external_capability(
            scanout_modifiers,
            width,
            height,
            OUTPUT_MODIFIER_FEATURES,
            vk::ExternalMemoryFeatureFlags::EXPORTABLE,
            "exportable",
        )?;

        if !result.is_empty() {
            return Ok(result);
        }

        // Fallback: let query_output_modifier pick the best one.
        let best = self.query_output_modifier()?;
        Ok(vec![best])
    }

    /// Compute modifiers that Vulkan can import from GBM-allocated DMA-BUFs.
    ///
    /// Returns modifiers that Vulkan reports as importable via
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`, intersected with
    /// `scanout_modifiers` when non-empty.
    pub fn compute_importable_modifiers(
        &self,
        scanout_modifiers: &[u64],
        width: u32,
        height: u32,
    ) -> anyhow::Result<Vec<u64>> {
        let result = self.filter_modifiers_by_external_capability(
            scanout_modifiers,
            width,
            height,
            IMPORTABLE_MODIFIER_FEATURES,
            vk::ExternalMemoryFeatureFlags::IMPORTABLE,
            "importable",
        )?;

        if !result.is_empty() {
            return Ok(result);
        }

        bail!("no importable modifiers found for GBM output path");
    }
}
