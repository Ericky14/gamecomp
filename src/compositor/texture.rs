//! Vulkan texture management.
//!
//! [`VulkanTexture`] wraps a `VkImage` imported from a client DMA-BUF.
//! Textures are created on the render thread and referenced by index
//! in the [`FrameInfo`] scene description. They are not shared across threads.
//!
//! Design: Textures are stored in a slab allocator indexed by `u32`.
//! The main thread refers to them by index only — it never touches Vulkan objects.

use ash::vk;

/// A Vulkan texture imported from a DMA-BUF.
#[derive(Debug)]
pub struct VulkanTexture {
    /// Vulkan image handle.
    pub image: vk::Image,
    /// Image view for sampling.
    pub view: vk::ImageView,
    /// Device memory backing the image (imported from DMA-BUF).
    pub memory: vk::DeviceMemory,
    /// Image dimensions.
    pub width: u32,
    pub height: u32,
    /// DRM format of the original buffer.
    pub drm_format: u32,
    /// Whether this texture slot is currently in use.
    pub in_use: bool,
}

impl Default for VulkanTexture {
    fn default() -> Self {
        Self {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            memory: vk::DeviceMemory::null(),
            width: 0,
            height: 0,
            drm_format: 0,
            in_use: false,
        }
    }
}

/// Fixed-size texture pool.
///
/// Pre-allocates slots to avoid per-frame allocation. Textures are reused
/// when clients destroy and recreate buffers.
pub const MAX_TEXTURES: usize = 32;

/// A pool of Vulkan textures, indexed by `u32`.
pub struct TexturePool {
    textures: [VulkanTexture; MAX_TEXTURES],
    /// Number of textures currently in use.
    count: u32,
}

impl TexturePool {
    /// Create a new empty texture pool.
    pub fn new() -> Self {
        Self {
            textures: std::array::from_fn(|_| VulkanTexture::default()),
            count: 0,
        }
    }

    /// Allocate a texture slot. Returns the index, or `None` if full.
    pub fn allocate(&mut self) -> Option<u32> {
        for (i, tex) in self.textures.iter_mut().enumerate() {
            if !tex.in_use {
                tex.in_use = true;
                self.count += 1;
                return Some(i as u32);
            }
        }
        None
    }

    /// Release a texture slot by index.
    ///
    /// # Safety
    ///
    /// The caller must ensure the Vulkan resources (image, view, memory) have
    /// been properly destroyed before calling this. The slot is marked as free
    /// and may be reused immediately.
    pub unsafe fn release(&mut self, index: u32) {
        // SAFETY: index is bounds-checked below.
        if (index as usize) < MAX_TEXTURES {
            let tex = &mut self.textures[index as usize];
            tex.in_use = false;
            tex.image = vk::Image::null();
            tex.view = vk::ImageView::null();
            tex.memory = vk::DeviceMemory::null();
            self.count -= 1;
        }
    }

    /// Get a texture by index.
    #[inline(always)]
    pub fn get(&self, index: u32) -> Option<&VulkanTexture> {
        self.textures.get(index as usize).filter(|t| t.in_use)
    }

    /// Get a mutable texture by index.
    #[inline(always)]
    pub fn get_mut(&mut self, index: u32) -> Option<&mut VulkanTexture> {
        self.textures.get_mut(index as usize).filter(|t| t.in_use)
    }

    /// Number of textures currently allocated.
    #[inline(always)]
    pub fn len(&self) -> u32 {
        self.count
    }

    /// Whether the pool is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_release() {
        let mut pool = TexturePool::new();
        let idx = pool.allocate().unwrap();
        assert_eq!(idx, 0);
        assert_eq!(pool.len(), 1);

        // SAFETY: No Vulkan resources to destroy in test.
        unsafe { pool.release(idx) };
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn pool_capacity() {
        let mut pool = TexturePool::new();
        for i in 0..MAX_TEXTURES {
            assert!(pool.allocate().is_some(), "failed to allocate slot {}", i);
        }
        assert!(pool.allocate().is_none(), "should be full");
    }

    #[test]
    fn reuse_released_slot() {
        let mut pool = TexturePool::new();
        let idx = pool.allocate().unwrap();
        // SAFETY: No Vulkan resources to destroy in test.
        unsafe { pool.release(idx) };
        let idx2 = pool.allocate().unwrap();
        assert_eq!(idx, idx2, "should reuse the released slot");
    }
}
