#include "HealthBarDetector.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace gameagent {
namespace {

constexpr int kCanonicalWidth = 512;
constexpr int kCanonicalHeight = 448;

struct Region {
    int x_begin;
    int x_end;
    int y_begin;
    int y_end;
};

int scaled_coordinate(int value, int actual, int canonical) {
    return std::clamp(
        static_cast<int>(std::lround(
            static_cast<double>(value) * actual / canonical
        )),
        0,
        actual
    );
}

Region scaled_region(
    int x_begin,
    int x_end,
    int y_begin,
    int y_end,
    int width,
    int height
) {
    Region region{
        .x_begin = scaled_coordinate(x_begin, width, kCanonicalWidth),
        .x_end = scaled_coordinate(x_end, width, kCanonicalWidth),
        .y_begin = scaled_coordinate(y_begin, height, kCanonicalHeight),
        .y_end = scaled_coordinate(y_end, height, kCanonicalHeight),
    };
    region.x_end = std::max(region.x_end, region.x_begin + 1);
    region.y_end = std::max(region.y_end, region.y_begin + 1);
    return region;
}

struct MeasuredBar {
    float health = 0.0f;
    float fill = 0.0f;
    std::size_t columns = 0;
};

MeasuredBar measure_bar(
    const base::data::image::Image& image,
    const Region& region,
    const HealthBarDetectorConfig& config
) {
    std::size_t blue_columns = 0;
    std::size_t filled_columns = 0;
    const int rows = region.y_end - region.y_begin;
    const int columns = region.x_end - region.x_begin;

    for (int x = region.x_begin; x < region.x_end; ++x) {
        std::size_t blue_pixels = 0;
        std::size_t red_pixels = 0;
        for (int y = region.y_begin; y < region.y_end; ++y) {
            const float red = image.at(y, x, 0);
            const float green = image.at(y, x, 1);
            const float blue = image.at(y, x, 2);
            const bool is_blue =
                blue > 150.0f &&
                blue > red * 1.80f &&
                blue > green * 1.80f;
            const bool is_red =
                red > 55.0f &&
                red > green * 1.50f &&
                red > blue * 1.20f;
            blue_pixels += is_blue ? 1U : 0U;
            red_pixels += is_red ? 1U : 0U;
        }
        const float blue_fraction =
            static_cast<float>(blue_pixels) / static_cast<float>(rows);
        const float red_fraction =
            static_cast<float>(red_pixels) / static_cast<float>(rows);
        const bool blue_column =
            blue_fraction >= config.blue_column_fraction;
        const bool red_column =
            red_fraction >= config.red_column_fraction;
        blue_columns += blue_column ? 1U : 0U;
        filled_columns += (blue_column || red_column) ? 1U : 0U;
    }

    return {
        .health = static_cast<float>(blue_columns) /
            static_cast<float>(columns),
        .fill = static_cast<float>(filled_columns) /
            static_cast<float>(columns),
        .columns = static_cast<std::size_t>(columns),
    };
}

float selected_health(
    const HealthBarReading& reading,
    PlayerSide side
) {
    return side == PlayerSide::left
        ? reading.left_health
        : reading.right_health;
}

float other_health(
    const HealthBarReading& reading,
    PlayerSide side
) {
    return side == PlayerSide::left
        ? reading.right_health
        : reading.left_health;
}

}  // namespace

std::string_view fight_phase_name(FightPhase phase) {
    switch (phase) {
        case FightPhase::no_hud: return "no_hud";
        case FightPhase::acquiring: return "acquiring";
        case FightPhase::round_intro: return "round_intro";
        case FightPhase::active_fight: return "active_fight";
        case FightPhase::round_end: return "round_end";
    }
    return "unknown";
}

HealthBarDetector::HealthBarDetector(HealthBarDetectorConfig config)
    : config_(config) {
    if (config_.minimum_fill_ratio <= 0.0f ||
        config_.minimum_fill_ratio > 1.0f ||
        config_.blue_column_fraction <= 0.0f ||
        config_.blue_column_fraction > 1.0f ||
        config_.red_column_fraction <= 0.0f ||
        config_.red_column_fraction > 1.0f) {
        throw std::invalid_argument("invalid health-bar detector configuration");
    }
}

HealthBarReading HealthBarDetector::measure(
    const base::data::image::Image& image
) const {
    if (image.empty() || image.channels() != 3 ||
        image.width() < 128 || image.height() < 112) {
        return {};
    }

    const Region left = scaled_region(
        31, 225, 39, 54, image.width(), image.height()
    );
    const Region right = scaled_region(
        287, 481, 39, 54, image.width(), image.height()
    );
    const MeasuredBar left_bar = measure_bar(image, left, config_);
    const MeasuredBar right_bar = measure_bar(image, right, config_);
    const float confidence = std::min(left_bar.fill, right_bar.fill);
    return {
        .hud_valid =
            left_bar.fill >= config_.minimum_fill_ratio &&
            right_bar.fill >= config_.minimum_fill_ratio,
        .left_health = left_bar.health,
        .right_health = right_bar.health,
        .left_fill_ratio = left_bar.fill,
        .right_fill_ratio = right_bar.fill,
        .confidence = confidence,
        .bar_columns = std::min(left_bar.columns, right_bar.columns),
    };
}

FightStateTracker::FightStateTracker(
    FightStateConfig config,
    HealthBarDetectorConfig detector_config
) : config_(config), detector_(detector_config) {
    if (config_.valid_frames_to_acquire == 0 ||
        config_.invalid_frames_to_end == 0 ||
        config_.stable_intro_frames == 0 ||
        config_.terminal_frames == 0 ||
        config_.round_intro_grace_ms < 0 ||
        config_.maximum_action_frame_age_ms <= 0 ||
        config_.health_increase_tolerance < 0.0f ||
        config_.damage_noise_tolerance < 0.0f ||
        config_.terminal_health_threshold < 0.0f ||
        config_.damage_dealt_reward_scale <= 0.0f ||
        config_.damage_taken_penalty_scale < 0.0f) {
        throw std::invalid_argument("invalid fight-state tracker configuration");
    }
}

void FightStateTracker::begin_active_episode_locked(
    float player_health,
    float opponent_health,
    std::int64_t timestamp_ms,
    std::string reason
) {
    ++latest_.episode_id;
    latest_.phase = FightPhase::active_fight;
    latest_.action_allowed = true;
    latest_.episode_started = true;
    latest_.transition_reason = reason;
    filtered_player_health_ = player_health;
    filtered_opponent_health_ = opponent_health;
    episode_active_ = true;
    terminal_frames_ = 0;
    pending_episode_started_ = true;
    pending_transition_reason_ = std::move(reason);
    static_cast<void>(timestamp_ms);
}

void FightStateTracker::end_episode_locked(
    bool terminal,
    std::string reason
) {
    if (episode_active_) {
        pending_episode_ended_ = true;
        pending_terminal_ = pending_terminal_ || terminal;
    }
    episode_active_ = false;
    awaiting_opponent_refill_ = false;
    latest_.action_allowed = false;
    latest_.episode_ended = true;
    latest_.terminal = terminal;
    latest_.transition_reason = reason;
    pending_transition_reason_ = std::move(reason);
}

FightControlSnapshot FightStateTracker::observe(const BufferedFrame& frame) {
    const HealthBarReading reading = detector_.measure(frame.image);
    const std::int64_t timestamp_ms = frame.client_timestamp_ms > 0
        ? frame.client_timestamp_ms
        : frame.received_timestamp_ms;
    std::lock_guard lock(mutex_);

    latest_.frame_id = frame.frame_id;
    latest_.frame_timestamp_ms = timestamp_ms;
    latest_.frame_received_timestamp_ms = frame.received_timestamp_ms;
    latest_.hud_valid = reading.hud_valid;
    latest_.hud_confidence = reading.confidence;
    latest_.episode_started = false;
    latest_.episode_ended = false;
    latest_.terminal = false;
    latest_.damage_dealt = 0.0f;
    latest_.damage_taken = 0.0f;
    latest_.reward = 0.0f;
    latest_.terminal_reward = 0.0f;
    latest_.transition_reason.clear();

    if (!reading.hud_valid) {
        ++invalid_frames_;
        valid_frames_ = 0;
        intro_stable_frames_ = 0;
        latest_.action_allowed = false;
        if (invalid_frames_ >= config_.invalid_frames_to_end) {
            if (episode_active_) {
                end_episode_locked(false, "hud_lost");
            }
            latest_.phase = FightPhase::no_hud;
            has_previous_reading_ = false;
        }
        return latest_;
    }

    invalid_frames_ = 0;
    ++valid_frames_;
    const float player_health = selected_health(reading, config_.player_side);
    const float opponent_health = other_health(reading, config_.player_side);
    latest_.player_health = player_health;
    latest_.opponent_health = opponent_health;

    float previous_player = player_health;
    float previous_opponent = opponent_health;
    bool health_increased = false;
    bool player_health_increased = false;
    bool opponent_health_increased = false;
    if (has_previous_reading_) {
        previous_player = selected_health(previous_reading_, config_.player_side);
        previous_opponent = other_health(previous_reading_, config_.player_side);
        player_health_increased =
            player_health > previous_player + config_.health_increase_tolerance;
        opponent_health_increased =
            opponent_health > previous_opponent + config_.health_increase_tolerance;
        health_increased =
            player_health_increased || opponent_health_increased;
    }
    previous_reading_ = reading;
    has_previous_reading_ = true;

    if (latest_.phase == FightPhase::active_fight) {
        if (
            external_encounter_lifecycle_ &&
            awaiting_opponent_refill_ &&
            opponent_health_increased &&
            !player_health_increased
        ) {
            // The new Endurance opponent legitimately starts with a full bar.
            // Rebase the detector and do not generate damage/reward/terminal.
            filtered_opponent_health_ = opponent_health;
            filtered_player_health_ = std::min(
                filtered_player_health_, player_health
            );
            awaiting_opponent_refill_ = false;
            terminal_frames_ = 0;
            latest_.player_health = filtered_player_health_;
            latest_.opponent_health = filtered_opponent_health_;
            latest_.action_allowed = true;
            latest_.transition_reason =
                "endurance_opponent_health_refill_ignored";
            pending_transition_reason_ = latest_.transition_reason;
            return latest_;
        }
        if (health_increased) {
            end_episode_locked(false, "health_refill_or_state_load");
            latest_.phase = FightPhase::round_intro;
            last_intro_increase_ms_ = timestamp_ms;
            intro_stable_frames_ = 0;
            return latest_;
        }

        const float next_player = std::min(
            filtered_player_health_, player_health
        );
        const float next_opponent = std::min(
            filtered_opponent_health_, opponent_health
        );
        const float player_delta = filtered_player_health_ - next_player;
        const float opponent_delta = filtered_opponent_health_ - next_opponent;
        const float damage_taken =
            player_delta >= config_.damage_noise_tolerance
                ? player_delta
                : 0.0f;
        const float damage_dealt =
            opponent_delta >= config_.damage_noise_tolerance
                ? opponent_delta
                : 0.0f;
        filtered_player_health_ = next_player;
        filtered_opponent_health_ = next_opponent;
        latest_.player_health = filtered_player_health_;
        latest_.opponent_health = filtered_opponent_health_;
        latest_.damage_dealt = damage_dealt;
        latest_.damage_taken = damage_taken;
        latest_.reward =
            config_.damage_dealt_reward_scale * damage_dealt -
            config_.damage_taken_penalty_scale * damage_taken;
        pending_reward_ += latest_.reward;
        pending_damage_dealt_ += damage_dealt;
        pending_damage_taken_ += damage_taken;

        const bool player_empty =
            filtered_player_health_ <= config_.terminal_health_threshold;
        const bool opponent_empty =
            filtered_opponent_health_ <= config_.terminal_health_threshold;
        if (player_empty || opponent_empty) {
            ++terminal_frames_;
            latest_.action_allowed = false;
        } else {
            terminal_frames_ = 0;
            latest_.action_allowed = true;
        }
        if (terminal_frames_ >= config_.terminal_frames) {
            const bool won = opponent_empty && !player_empty;
            const bool lost = player_empty && !opponent_empty;
            // Endurance encounters keep their intermediate opponent
            // transitions under the explicit V2.1 lifecycle. A depleted
            // player bar is always terminal, though: waiting for a client
            // encounter_lost event used to turn almost every real loss into
            // a reward-less hud_lost recovery.
            if (external_encounter_lifecycle_ && !lost) {
                return latest_;
            }
            const float terminal_reward =
                won ? config_.round_win_bonus :
                lost ? -config_.round_loss_penalty : 0.0f;
            latest_.terminal_reward = terminal_reward;
            latest_.reward += terminal_reward;
            pending_reward_ += terminal_reward;
            pending_terminal_reward_ += terminal_reward;
            end_episode_locked(true, won ? "round_won" : lost ? "round_lost" : "double_ko");
            latest_.phase = FightPhase::round_end;
        }
        return latest_;
    }

    latest_.action_allowed = false;
    if (player_health <= config_.terminal_health_threshold ||
        opponent_health <= config_.terminal_health_threshold) {
        latest_.phase = FightPhase::round_end;
        return latest_;
    }
    if (latest_.phase == FightPhase::round_end) {
        if (!health_increased) {
            return latest_;
        }
        latest_.phase = FightPhase::round_intro;
        last_intro_increase_ms_ = timestamp_ms;
        intro_stable_frames_ = 0;
        latest_.transition_reason = "next_round_refill";
        pending_transition_reason_ = latest_.transition_reason;
        return latest_;
    }

    const bool both_nearly_full =
        player_health >= 0.97f && opponent_health >= 0.97f;
    if (latest_.phase == FightPhase::no_hud) {
        if (both_nearly_full) {
            latest_.phase = FightPhase::round_intro;
            last_intro_increase_ms_ = timestamp_ms;
            intro_stable_frames_ = 1;
        } else {
            latest_.phase = FightPhase::acquiring;
        }
    }

    if (health_increased) {
        latest_.phase = FightPhase::round_intro;
        last_intro_increase_ms_ = timestamp_ms;
        intro_stable_frames_ = 0;
        return latest_;
    }

    if (latest_.phase == FightPhase::round_intro) {
        ++intro_stable_frames_;
        const bool grace_complete =
            timestamp_ms - last_intro_increase_ms_ >=
                config_.round_intro_grace_ms;
        if (intro_stable_frames_ >= config_.stable_intro_frames &&
            grace_complete) {
            begin_active_episode_locked(
                player_health,
                opponent_health,
                timestamp_ms,
                "round_ready"
            );
        }
        return latest_;
    }

    if (latest_.phase == FightPhase::acquiring &&
        valid_frames_ >= config_.valid_frames_to_acquire) {
        begin_active_episode_locked(
            player_health,
            opponent_health,
            timestamp_ms,
            "mid_round_acquired"
        );
    }
    return latest_;
}

void FightStateTracker::set_external_encounter_lifecycle(bool enabled) {
    std::lock_guard lock(mutex_);
    external_encounter_lifecycle_ = enabled;
    awaiting_opponent_refill_ = false;
}

void FightStateTracker::advance_encounter_stage(float stage_bonus) {
    if (!std::isfinite(stage_bonus)) {
        throw std::invalid_argument("stage bonus must be finite");
    }
    std::lock_guard lock(mutex_);
    if (!external_encounter_lifecycle_) {
        throw std::logic_error(
            "cannot advance stage without an external encounter"
        );
    }
    pending_reward_ += stage_bonus;
    latest_.reward += stage_bonus;
    latest_.terminal = false;
    latest_.episode_ended = false;
    latest_.action_allowed = false;
    latest_.transition_reason = "endurance_stage_completed";
    pending_transition_reason_ = latest_.transition_reason;
    awaiting_opponent_refill_ = true;
    terminal_frames_ = 0;
}

void FightStateTracker::finish_encounter(bool won) {
    std::lock_guard lock(mutex_);
    if (!external_encounter_lifecycle_ || !episode_active_) {
        return;
    }
    const float terminal_reward = won
        ? config_.round_win_bonus
        : -config_.round_loss_penalty;
    pending_reward_ += terminal_reward;
    pending_terminal_reward_ += terminal_reward;
    latest_.reward += terminal_reward;
    latest_.terminal_reward = terminal_reward;
    end_episode_locked(true, won ? "encounter_won" : "encounter_lost");
    latest_.phase = FightPhase::round_end;
}

FightControlSnapshot FightStateTracker::make_snapshot_locked(
    std::int64_t now_ms
) const {
    FightControlSnapshot result = latest_;
    if (result.action_allowed &&
        (result.frame_received_timestamp_ms <= 0 ||
         now_ms - result.frame_received_timestamp_ms >
            config_.maximum_action_frame_age_ms)) {
        result.action_allowed = false;
        result.transition_reason = "stale_active_frame";
    }
    return result;
}

FightControlSnapshot FightStateTracker::consume(std::int64_t now_ms) {
    std::lock_guard lock(mutex_);
    FightControlSnapshot result = make_snapshot_locked(now_ms);
    result.reward = pending_reward_;
    result.terminal_reward = pending_terminal_reward_;
    result.damage_dealt = pending_damage_dealt_;
    result.damage_taken = pending_damage_taken_;
    result.episode_started = pending_episode_started_;
    result.episode_ended = pending_episode_ended_;
    result.terminal = pending_terminal_;
    if (!pending_transition_reason_.empty()) {
        result.transition_reason = pending_transition_reason_;
    }
    pending_reward_ = 0.0f;
    pending_terminal_reward_ = 0.0f;
    pending_damage_dealt_ = 0.0f;
    pending_damage_taken_ = 0.0f;
    pending_episode_started_ = false;
    pending_episode_ended_ = false;
    pending_terminal_ = false;
    pending_transition_reason_.clear();
    return result;
}

FightControlSnapshot FightStateTracker::snapshot(
    std::int64_t now_ms
) const {
    std::lock_guard lock(mutex_);
    FightControlSnapshot result = make_snapshot_locked(now_ms);
    result.reward = 0.0f;
    result.terminal_reward = 0.0f;
    result.damage_dealt = 0.0f;
    result.damage_taken = 0.0f;
    result.episode_started = false;
    result.episode_ended = false;
    result.terminal = false;
    return result;
}

void FightStateTracker::reset() {
    std::lock_guard lock(mutex_);
    latest_ = {};
    previous_reading_ = {};
    has_previous_reading_ = false;
    filtered_player_health_ = 0.0f;
    filtered_opponent_health_ = 0.0f;
    valid_frames_ = 0;
    invalid_frames_ = 0;
    intro_stable_frames_ = 0;
    terminal_frames_ = 0;
    last_intro_increase_ms_ = 0;
    episode_active_ = false;
    awaiting_opponent_refill_ = false;
    pending_reward_ = 0.0f;
    pending_terminal_reward_ = 0.0f;
    pending_damage_dealt_ = 0.0f;
    pending_damage_taken_ = 0.0f;
    pending_episode_started_ = false;
    pending_episode_ended_ = false;
    pending_terminal_ = false;
    pending_transition_reason_.clear();
}

TrainingResetTracker::TrainingResetTracker(
    bool reset_after_round_win,
    std::int64_t non_fight_timeout_ms,
    std::int64_t round_intro_timeout_ms
) : reset_after_round_win_(reset_after_round_win),
    non_fight_timeout_ms_(non_fight_timeout_ms),
    round_intro_timeout_ms_(round_intro_timeout_ms) {
    if (non_fight_timeout_ms_ <= 0 ||
        round_intro_timeout_ms_ < non_fight_timeout_ms_) {
        throw std::invalid_argument(
            "reset timeouts must be positive and round-intro timeout "
            "must not be shorter than the general non-fight timeout"
        );
    }
}

std::optional<TrainingResetTrigger> TrainingResetTracker::observe(
    const FightControlSnapshot& fight,
    std::int64_t now_ms
) {
    const bool reset_boundary =
        fight.transition_reason == "round_lost" ||
        (
            reset_after_round_win_ &&
            fight.transition_reason == "round_won"
        );
    if (
        fight.episode_ended && fight.terminal &&
        fight.episode_id != 0 && reset_boundary &&
        fight.episode_id != last_reset_episode_id_
    ) {
        last_reset_episode_id_ = fight.episode_id;
        non_fight_since_ms_ = now_ms;
        return TrainingResetTrigger{
            .episode_id = fight.episode_id,
            .frame_id = fight.frame_id,
            .reason = fight.transition_reason,
        };
    }

    if (fight.phase == FightPhase::active_fight) {
        non_fight_since_ms_.reset();
        return std::nullopt;
    }
    if (
        !non_fight_since_ms_.has_value() ||
        now_ms < *non_fight_since_ms_
    ) {
        non_fight_since_ms_ = now_ms;
        return std::nullopt;
    }
    const std::int64_t timeout_ms =
        fight.phase == FightPhase::round_intro
            ? round_intro_timeout_ms_
            : non_fight_timeout_ms_;
    if (now_ms - *non_fight_since_ms_ < timeout_ms) {
        return std::nullopt;
    }
    non_fight_since_ms_ = now_ms;
    return TrainingResetTrigger{
        .episode_id = fight.episode_id,
        .frame_id = fight.frame_id,
        .reason = "non_fight_timeout",
    };
}

void TrainingResetTracker::reset() {
    last_reset_episode_id_ = 0;
    non_fight_since_ms_.reset();
}

}  // namespace gameagent
