#pragma once

#include "ActionHistory.h"
#include "FrameBuffer.h"
#include "HealthBarDetector.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace gameagent {

// Four health/HUD values, ten confirmed controller values and the ACK-known
// flag were the original context. The remaining values make the fight state
// explicit instead of forcing the recurrent policy to rediscover every
// temporal/spatial relationship from pixels alone.
inline constexpr std::size_t kReinforcementContextFeatureCount = 30;
inline constexpr std::size_t kCombatStateFeatureCount = 15;

struct CombatStateConfig {
    PlayerSide player_side = PlayerSide::left;
    float motion_threshold = 30.0f;
    float minimum_spatial_confidence = 0.04f;
    float position_smoothing = 0.35f;
    float corner_boundary = 0.14f;
    float stationary_position_delta = 0.012f;
    std::int64_t combo_timeout_ms = 1200;
    std::int64_t attack_memory_ms = 500;
    std::int64_t idle_grace_ms = 1500;
    std::int64_t corner_grace_ms = 1000;
    std::int64_t maximum_reward_delta_ms = 250;
    std::int64_t time_feature_horizon_ms = 5000;
    float combo_reward_scale = 0.05f;
    float idle_penalty_per_second = 0.01f;
    float corner_penalty_per_second = 0.02f;
};

struct CombatStateSnapshot {
    std::uint64_t frame_id = 0;
    std::int64_t timestamp_ms = 0;
    bool spatial_valid = false;
    float spatial_confidence = 0.0f;
    float opponent_action_confidence = 0.0f;
    float player_x = 0.30f;
    float opponent_x = 0.70f;
    float distance = 0.40f;
    bool player_on_ground = true;
    bool opponent_on_ground = true;
    bool player_defending = false;
    bool opponent_defending = false;
    bool player_attacking = false;
    bool opponent_attacking = false;
    std::int64_t time_since_last_damage_ms = 5000;
    std::size_t consecutive_hits = 0;
    std::int64_t stationary_time_ms = 0;
    std::int64_t corner_time_ms = 0;
    float shaping_reward = 0.0f;
    float combo_reward = 0.0f;
    float idle_penalty = 0.0f;
    float corner_penalty = 0.0f;
};

// Lightweight screen-space tracker for the fixed 512x448-style game capture.
// Motion is measured only below the HUD, associated with the two previous
// fighter positions and smoothed over time. Uncertain estimates are retained
// with an explicit confidence rather than fabricated as hard ground truth.
class CombatStateTracker {
public:
    explicit CombatStateTracker(CombatStateConfig config = {});

    CombatStateSnapshot observe(
        const BufferedFrame& frame,
        const FightControlSnapshot& fight,
        const std::optional<AppliedActionRecord>& latest_applied
    );
    CombatStateSnapshot consume(std::int64_t now_ms);
    [[nodiscard]] CombatStateSnapshot snapshot(
        std::int64_t now_ms
    ) const;
    void reset();

private:
    struct MotionTrack {
        bool valid = false;
        float x = 0.0f;
        float bottom_y = 0.86f;
        float motion = 0.0f;
        float confidence = 0.0f;
    };

    std::vector<float> grayscale(const BufferedFrame& frame) const;
    std::pair<MotionTrack, MotionTrack> measure_tracks(
        const BufferedFrame& frame,
        const std::vector<float>& current_gray
    ) const;
    CombatStateSnapshot make_snapshot_locked(
        std::int64_t now_ms
    ) const;
    void reset_episode_locked(std::int64_t timestamp_ms);

    CombatStateConfig config_;
    mutable std::mutex mutex_;
    CombatStateSnapshot latest_;
    std::vector<float> previous_gray_;
    int previous_width_ = 0;
    int previous_height_ = 0;
    float tracked_player_x_ = 0.30f;
    float tracked_opponent_x_ = 0.70f;
    float player_bottom_y_ = 0.86f;
    float opponent_bottom_y_ = 0.86f;
    float player_tracking_confidence_ = 0.0f;
    float opponent_tracking_confidence_ = 0.0f;
    bool tracking_initialized_ = false;
    std::int64_t previous_timestamp_ms_ = 0;
    std::int64_t last_damage_ms_ = 0;
    std::int64_t last_damage_dealt_ms_ = 0;
    std::int64_t last_damage_taken_ms_ = 0;
    std::int64_t last_player_attack_ms_ = 0;
    std::int64_t player_airborne_until_ms_ = 0;
    std::int64_t stationary_started_ms_ = 0;
    std::int64_t corner_started_ms_ = 0;
    std::size_t consecutive_hits_ = 0;
    float pending_shaping_reward_ = 0.0f;
    float pending_combo_reward_ = 0.0f;
    float pending_idle_penalty_ = 0.0f;
    float pending_corner_penalty_ = 0.0f;
};

}  // namespace gameagent
