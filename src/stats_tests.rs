use super::*;

#[test]
fn stats_tracker_new_defaults() {
    let tracker = StatsTracker::new(None);
    assert_eq!(tracker.total_frames(), 0);
    assert_eq!(tracker.fps(), 0.0);
    assert_eq!(tracker.avg_draw_time_ns(), 0);
}

#[test]
fn stats_tracker_records_frames() {
    let mut tracker = StatsTracker::new(None);
    let stats = FrameStats {
        seq: 1,
        draw_time_ns: 5_000_000,
        flip_time_ns: 1_000_000,
        direct_scanout: false,
        vrr_active: false,
        layer_count: 2,
        fps: 60.0,
    };
    tracker.record_frame(&stats);
    assert_eq!(tracker.total_frames(), 1);
}

#[test]
fn stats_tracker_avg_draw_time() {
    let mut tracker = StatsTracker::new(None);

    for i in 1..=10 {
        let stats = FrameStats {
            seq: i,
            draw_time_ns: 4_000_000, // 4ms constant.
            ..Default::default()
        };
        tracker.record_frame(&stats);
    }

    assert_eq!(tracker.avg_draw_time_ns(), 4_000_000);
}

#[test]
fn stats_tracker_ring_buffer_wraps() {
    let mut tracker = StatsTracker::new(None);

    // Fill the ring buffer and wrap around.
    for i in 0..120 {
        let stats = FrameStats {
            seq: i,
            draw_time_ns: 3_000_000,
            ..Default::default()
        };
        tracker.record_frame(&stats);
    }

    assert_eq!(tracker.total_frames(), 120);
    assert_eq!(tracker.avg_draw_time_ns(), 3_000_000);
}

#[test]
fn frame_stats_default() {
    let stats = FrameStats::default();
    assert_eq!(stats.seq, 0);
    assert_eq!(stats.draw_time_ns, 0);
    assert!(!stats.direct_scanout);
    assert!(!stats.vrr_active);
    assert_eq!(stats.layer_count, 0);
}
