#pragma once

#include "FrameBuffer.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace gameagent {

enum class PlayerSide {
    left,
    right,
};

enum class FightPhase {
    no_hud,
    acquiring,
    round_intro,
    active_fight,
    round_end,
};

std::string_view fight_phase_name(FightPhase phase);

struct HealthBarReading {
    bool hud_valid = false;
    float left_health = 0.0f;
    float right_health = 0.0f;
    float left_fill_ratio = 0.0f;
    float right_fill_ratio = 0.0f;
    float confidence = 0.0f;
    std::size_t bar_columns = 0;
};

struct HealthBarDetectorConfig {
    float minimum_fill_ratio = 0.90f;
    float blue_column_fraction = 0.55f;
    float red_column_fraction = 0.18f;
};

// Fast deterministic detector for the fixed Mortal Kombat HUD. Coordinates
// are scaled from the canonical 512x448 capture, so 511x446 and other nearby
// resolutions use the same normalized regions.
class HealthBarDetector {
public:
    explicit HealthBarDetector(HealthBarDetectorConfig config = {});

    [[nodiscard]] HealthBarReading measure(
        const base::data::image::Image& image
    ) const;

private:
    HealthBarDetectorConfig config_;
};

struct FightStateConfig {
    PlayerSide player_side = PlayerSide::left;
    std::size_t valid_frames_to_acquire = 3;
    std::size_t invalid_frames_to_end = 2;
    std::size_t stable_intro_frames = 3;
    std::size_t terminal_frames = 2;
    std::int64_t round_intro_grace_ms = 1500;
    std::int64_t maximum_action_frame_age_ms = 250;
    float health_increase_tolerance = 0.015f;
    float damage_noise_tolerance = 0.0075f;
    float terminal_health_threshold = 0.010f;
    float damage_dealt_reward_scale = 15.0f;
    float damage_taken_penalty_scale = 11.0f;
    float round_win_bonus = 5.0f;
    float round_loss_penalty = 5.0f;
};

struct FightControlSnapshot {
    std::uint64_t frame_id = 0;
    std::int64_t frame_timestamp_ms = 0;
    std::int64_t frame_received_timestamp_ms = 0;
    std::uint64_t episode_id = 0;
    FightPhase phase = FightPhase::no_hud;
    bool hud_valid = false;
    bool action_allowed = false;
    bool episode_started = false;
    bool episode_ended = false;
    bool terminal = false;
    float player_health = 0.0f;
    float opponent_health = 0.0f;
    float damage_dealt = 0.0f;
    float damage_taken = 0.0f;
    float reward = 0.0f;
    float terminal_reward = 0.0f;
    float hud_confidence = 0.0f;
    std::string transition_reason;
};

struct TrainingResetTrigger {
    std::uint64_t episode_id = 0;
    std::uint64_t frame_id = 0;
    std::string reason;
};

// Training resets failed attempts after a loss. Read-only evaluation may also
// reset after a win. A non-fight watchdog recovers menus, loading screens and
// missed round boundaries. Terminal boundaries produce at most one request;
// a persistent non-fight state may request recovery again after each timeout.
class TrainingResetTracker {
public:
    explicit TrainingResetTracker(
        bool reset_after_round_win = false,
        std::int64_t non_fight_timeout_ms = 5000
    );
    [[nodiscard]] std::optional<TrainingResetTrigger> observe(
        const FightControlSnapshot& fight,
        std::int64_t now_ms
    );
    void reset();

private:
    bool reset_after_round_win_ = false;
    std::int64_t non_fight_timeout_ms_ = 5000;
    std::optional<std::int64_t> non_fight_since_ms_;
    std::uint64_t last_reset_episode_id_ = 0;
};

// Converts individual bar measurements into an episode/control state. Reward
// and boundary flags are accumulated until consume() so asynchronous camera
// and controller loops cannot double-count the same damage.
class FightStateTracker {
public:
    explicit FightStateTracker(
        FightStateConfig config = {},
        HealthBarDetectorConfig detector_config = {}
    );

    FightControlSnapshot observe(const BufferedFrame& frame);
    FightControlSnapshot consume(std::int64_t now_ms);
    [[nodiscard]] FightControlSnapshot snapshot(std::int64_t now_ms) const;
    void reset();

private:
    FightControlSnapshot make_snapshot_locked(std::int64_t now_ms) const;
    void begin_active_episode_locked(
        float player_health,
        float opponent_health,
        std::int64_t timestamp_ms,
        std::string reason
    );
    void end_episode_locked(bool terminal, std::string reason);

    FightStateConfig config_;
    HealthBarDetector detector_;
    mutable std::mutex mutex_;
    FightControlSnapshot latest_;
    HealthBarReading previous_reading_;
    bool has_previous_reading_ = false;
    float filtered_player_health_ = 0.0f;
    float filtered_opponent_health_ = 0.0f;
    std::size_t valid_frames_ = 0;
    std::size_t invalid_frames_ = 0;
    std::size_t intro_stable_frames_ = 0;
    std::size_t terminal_frames_ = 0;
    std::int64_t last_intro_increase_ms_ = 0;
    bool episode_active_ = false;
    float pending_reward_ = 0.0f;
    float pending_terminal_reward_ = 0.0f;
    float pending_damage_dealt_ = 0.0f;
    float pending_damage_taken_ = 0.0f;
    bool pending_episode_started_ = false;
    bool pending_episode_ended_ = false;
    bool pending_terminal_ = false;
    std::string pending_transition_reason_;
};

}  // namespace gameagent
