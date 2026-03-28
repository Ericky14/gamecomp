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
