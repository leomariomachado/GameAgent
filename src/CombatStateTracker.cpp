#include "CombatStateTracker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gameagent {
namespace {

constexpr std::size_t kAttackBegin = 0;
constexpr std::size_t kAttackEnd = 4;
constexpr std::size_t kBlockCommand = 4;
constexpr std::size_t kUpCommand = 6;

float clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

bool command_active(
    const std::optional<AppliedActionRecord>& action,
    std::size_t command
) {
    return action.has_value() &&
        command < action->command_state.size() &&
        action->command_state[command] > 0.5f;
}

bool attack_active(
    const std::optional<AppliedActionRecord>& action
) {
    if (!action.has_value()) {
        return false;
    }
    return std::any_of(
        action->command_state.begin() +
            static_cast<std::ptrdiff_t>(kAttackBegin),
        action->command_state.begin() +
            static_cast<std::ptrdiff_t>(kAttackEnd),
        [](float value) { return value > 0.5f; }
    );
}

bool any_command_active(
    const std::optional<AppliedActionRecord>& action
) {
    return action.has_value() &&
        std::any_of(
            action->command_state.begin(),
            action->command_state.end(),
            [](float value) { return value > 0.5f; }
        );
}

float exponential_motion_score(float distance) {
    return std::exp(-distance / 0.20f);
}

}  // namespace

CombatStateTracker::CombatStateTracker(CombatStateConfig config)
    : config_(config) {
    if (
        config_.motion_threshold <= 0.0f ||
        config_.minimum_spatial_confidence < 0.0f ||
        config_.minimum_spatial_confidence > 1.0f ||
        config_.position_smoothing <= 0.0f ||
        config_.position_smoothing > 1.0f ||
        config_.corner_boundary <= 0.0f ||
        config_.corner_boundary >= 0.5f ||
        config_.stationary_position_delta <= 0.0f ||
        config_.combo_timeout_ms <= 0 ||
        config_.attack_memory_ms <= 0 ||
        config_.idle_grace_ms < 0 ||
        config_.corner_grace_ms < 0 ||
        config_.maximum_reward_delta_ms <= 0 ||
        config_.time_feature_horizon_ms <= 0 ||
        config_.combo_reward_scale < 0.0f ||
        config_.idle_penalty_per_second < 0.0f ||
        config_.corner_penalty_per_second < 0.0f
    ) {
        throw std::invalid_argument(
            "invalid combat-state tracker configuration"
        );
    }
    reset();
}

std::vector<float> CombatStateTracker::grayscale(
    const BufferedFrame& frame
) const {
    const auto& image = frame.image;
    if (
        image.empty() ||
        image.channels() != 3 ||
        image.width() <= 0 ||
        image.height() <= 0
    ) {
        return {};
    }
    std::vector<float> gray(
        static_cast<std::size_t>(image.width()) *
        static_cast<std::size_t>(image.height())
    );
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const float red = image.at(y, x, 0);
            const float green = image.at(y, x, 1);
            const float blue = image.at(y, x, 2);
            gray[
                static_cast<std::size_t>(y) * image.width() +
                static_cast<std::size_t>(x)
            ] = 0.299f * red + 0.587f * green + 0.114f * blue;
        }
    }
    return gray;
}

std::pair<
    CombatStateTracker::MotionTrack,
    CombatStateTracker::MotionTrack
> CombatStateTracker::measure_tracks(
    const BufferedFrame& frame,
    const std::vector<float>& current_gray
) const {
    const int width = frame.image.width();
    const int height = frame.image.height();
    if (
        previous_gray_.size() != current_gray.size() ||
        previous_width_ != width ||
        previous_height_ != height ||
        width < 128 ||
        height < 112
    ) {
        return {};
    }

    const int y_begin = std::clamp(
        static_cast<int>(std::lround(height * 0.16)), 0, height - 1
    );
    const int y_end = std::clamp(
        static_cast<int>(std::lround(height * 0.92)),
        y_begin + 1,
        height
    );
    const int x_margin = std::max(2, width / 50);
    std::vector<float> column_motion(
        static_cast<std::size_t>(width), 0.0f
    );
    std::vector<int> column_bottom(
        static_cast<std::size_t>(width), -1
    );
    std::size_t total_motion_pixels = 0;
    for (int y = y_begin; y < y_end; y += 2) {
        const std::size_t row =
            static_cast<std::size_t>(y) * width;
        for (int x = x_margin; x < width - x_margin; x += 2) {
            const std::size_t index = row + static_cast<std::size_t>(x);
            const float difference = std::abs(
                current_gray[index] - previous_gray_[index]
            );
            if (difference < config_.motion_threshold) {
                continue;
            }
            column_motion[static_cast<std::size_t>(x)] += 1.0f;
            column_bottom[static_cast<std::size_t>(x)] = y;
            ++total_motion_pixels;
        }
    }
    if (total_motion_pixels < 24U) {
        return {};
    }

    const int radius = std::max(6, width / 24);
    std::vector<float> smooth(
        static_cast<std::size_t>(width), 0.0f
    );
    float running = 0.0f;
    for (int x = 0; x < width; ++x) {
        running += column_motion[static_cast<std::size_t>(x)];
        if (x - radius >= 0) {
            running -= column_motion[
                static_cast<std::size_t>(x - radius)
            ];
        }
        smooth[static_cast<std::size_t>(x)] = running;
    }

    const auto locate = [&](float previous_x) {
        const int previous_pixel = std::clamp(
            static_cast<int>(std::lround(previous_x * (width - 1))),
            0,
            width - 1
        );
        const int search_radius = std::max(width / 5, 32);
        const int begin = std::max(x_margin, previous_pixel - search_radius);
        const int end = std::min(
            width - x_margin, previous_pixel + search_radius + 1
        );
        int peak = previous_pixel;
        float peak_score = 0.0f;
        for (int x = begin; x < end; ++x) {
            const float normalized_distance =
                std::abs(x - previous_pixel) /
                static_cast<float>(std::max(1, width - 1));
            const float score =
                smooth[static_cast<std::size_t>(x)] *
                exponential_motion_score(normalized_distance);
            if (score > peak_score) {
                peak_score = score;
                peak = x;
            }
        }

        const int local_radius = std::max(8, width / 14);
        const int local_begin = std::max(x_margin, peak - local_radius);
        const int local_end = std::min(
            width - x_margin, peak + local_radius + 1
        );
        double weighted_x = 0.0;
        double weight = 0.0;
        int bottom = -1;
        for (int x = local_begin; x < local_end; ++x) {
            const float activity =
                column_motion[static_cast<std::size_t>(x)];
            weighted_x += static_cast<double>(x) * activity;
            weight += activity;
            bottom = std::max(
                bottom,
                column_bottom[static_cast<std::size_t>(x)]
            );
        }
        const float normalized_height =
            static_cast<float>(std::max(1, y_end - y_begin));
        const float normalized_width =
            static_cast<float>(std::max(1, local_end - local_begin));
        const float confidence = clamp01(
            static_cast<float>(weight) /
            (0.18f * normalized_height * normalized_width)
        );
        return MotionTrack{
            .valid =
                weight > 0.0 &&
                confidence >= config_.minimum_spatial_confidence,
            .x = weight > 0.0
                ? clamp01(
                    static_cast<float>(weighted_x / weight) /
                    static_cast<float>(std::max(1, width - 1))
                )
                : previous_x,
            .bottom_y = bottom >= 0
                ? clamp01(
                    static_cast<float>(bottom) /
                    static_cast<float>(std::max(1, height - 1))
                )
                : 0.0f,
            .motion = clamp01(
                peak_score /
                std::max(1.0f, normalized_height * 0.12f)
            ),
            .confidence = confidence,
        };
    };

    MotionTrack player = locate(tracked_player_x_);
    MotionTrack opponent = locate(tracked_opponent_x_);
    if (
        player.valid &&
        opponent.valid &&
        std::abs(player.x - opponent.x) < 0.08f
    ) {
        if (player.confidence >= opponent.confidence) {
            opponent.valid = false;
            opponent.confidence *= 0.5f;
        } else {
            player.valid = false;
            player.confidence *= 0.5f;
        }
    }
    return {player, opponent};
}

void CombatStateTracker::reset_episode_locked(
    std::int64_t timestamp_ms
) {
    tracked_player_x_ =
        config_.player_side == PlayerSide::left ? 0.30f : 0.70f;
    tracked_opponent_x_ = 1.0f - tracked_player_x_;
    player_bottom_y_ = 0.86f;
    opponent_bottom_y_ = 0.86f;
    player_tracking_confidence_ = 0.0f;
    opponent_tracking_confidence_ = 0.0f;
    tracking_initialized_ = false;
    previous_gray_.clear();
    previous_width_ = 0;
    previous_height_ = 0;
    previous_timestamp_ms_ = timestamp_ms;
    last_damage_ms_ = 0;
    last_damage_dealt_ms_ = 0;
    last_damage_taken_ms_ = 0;
    last_player_attack_ms_ = 0;
    player_airborne_until_ms_ = 0;
    stationary_started_ms_ = timestamp_ms;
    corner_started_ms_ = 0;
    consecutive_hits_ = 0;
    latest_.stationary_time_ms = 0;
    latest_.corner_time_ms = 0;
}

CombatStateSnapshot CombatStateTracker::observe(
    const BufferedFrame& frame,
    const FightControlSnapshot& fight,
    const std::optional<AppliedActionRecord>& latest_applied
) {
    const std::int64_t timestamp_ms =
        frame.received_timestamp_ms > 0
            ? frame.received_timestamp_ms
            : frame.client_timestamp_ms;
    std::vector<float> current_gray = grayscale(frame);
    std::lock_guard lock(mutex_);

    if (fight.episode_started) {
        reset_episode_locked(timestamp_ms);
    }
    const auto [player_track, opponent_track] =
        measure_tracks(frame, current_gray);
    const float previous_player_x = tracked_player_x_;
    if (player_track.valid) {
        tracked_player_x_ =
            (1.0f - config_.position_smoothing) * tracked_player_x_ +
            config_.position_smoothing * player_track.x;
        player_bottom_y_ =
            (1.0f - config_.position_smoothing) * player_bottom_y_ +
            config_.position_smoothing * player_track.bottom_y;
        player_tracking_confidence_ = std::max(
            player_track.confidence,
            player_tracking_confidence_ * 0.90f
        );
        tracking_initialized_ = true;
    } else {
        player_tracking_confidence_ *= 0.97f;
    }
    if (opponent_track.valid) {
        tracked_opponent_x_ =
            (1.0f - config_.position_smoothing) * tracked_opponent_x_ +
            config_.position_smoothing * opponent_track.x;
        opponent_bottom_y_ =
            (1.0f - config_.position_smoothing) * opponent_bottom_y_ +
            config_.position_smoothing * opponent_track.bottom_y;
        opponent_tracking_confidence_ = std::max(
            opponent_track.confidence,
            opponent_tracking_confidence_ * 0.90f
        );
        tracking_initialized_ = true;
    } else {
        opponent_tracking_confidence_ *= 0.97f;
    }

    const std::int64_t delta_ms = previous_timestamp_ms_ > 0
        ? std::clamp(
              timestamp_ms - previous_timestamp_ms_,
              std::int64_t{0},
              config_.maximum_reward_delta_ms
          )
        : 0;
    previous_timestamp_ms_ = timestamp_ms;
    previous_width_ = frame.image.width();
    previous_height_ = frame.image.height();
    previous_gray_ = std::move(current_gray);

    const bool confirmed_attack = attack_active(latest_applied);
    const bool confirmed_block =
        command_active(latest_applied, kBlockCommand);
    if (confirmed_attack || fight.damage_dealt > 0.0f) {
        last_player_attack_ms_ = timestamp_ms;
    }
    if (command_active(latest_applied, kUpCommand)) {
        player_airborne_until_ms_ =
            timestamp_ms + std::max<std::int64_t>(
                config_.attack_memory_ms, 550
            );
    }

    float combo_reward = 0.0f;
    if (fight.damage_dealt > 0.0f) {
        if (
            last_damage_dealt_ms_ > 0 &&
            timestamp_ms - last_damage_dealt_ms_ <=
                config_.combo_timeout_ms
        ) {
            ++consecutive_hits_;
        } else {
            consecutive_hits_ = 1;
        }
        last_damage_dealt_ms_ = timestamp_ms;
        last_damage_ms_ = timestamp_ms;
        if (consecutive_hits_ > 1) {
            combo_reward = config_.combo_reward_scale *
                static_cast<float>(
                    std::min<std::size_t>(
                        consecutive_hits_ - 1U, 5U
                    )
                );
        }
    }
    if (fight.damage_taken > 0.0f) {
        last_damage_taken_ms_ = timestamp_ms;
        last_damage_ms_ = timestamp_ms;
        consecutive_hits_ = 0;
    } else if (
        last_damage_dealt_ms_ > 0 &&
        timestamp_ms - last_damage_dealt_ms_ >
            config_.combo_timeout_ms
    ) {
        consecutive_hits_ = 0;
    }

    const float player_delta =
        std::abs(tracked_player_x_ - previous_player_x);
    const bool player_visually_active =
        player_track.valid &&
        (
            player_delta >= config_.stationary_position_delta ||
            player_track.motion >= 0.10f
        );
    const bool player_active =
        player_visually_active ||
        any_command_active(latest_applied) ||
        fight.damage_dealt > 0.0f ||
        fight.damage_taken > 0.0f;
    if (
        !fight.action_allowed ||
        player_active ||
        stationary_started_ms_ == 0
    ) {
        stationary_started_ms_ = timestamp_ms;
    }
    const std::int64_t stationary_ms =
        fight.action_allowed
            ? std::max<std::int64_t>(
                  0, timestamp_ms - stationary_started_ms_
              )
            : 0;

    const bool in_corner =
        player_tracking_confidence_ >=
            config_.minimum_spatial_confidence &&
        (
        tracked_player_x_ <= config_.corner_boundary ||
        tracked_player_x_ >= 1.0f - config_.corner_boundary
        );
    if (!fight.action_allowed || !in_corner) {
        corner_started_ms_ = 0;
    } else if (corner_started_ms_ == 0) {
        corner_started_ms_ = timestamp_ms;
    }
    const std::int64_t corner_ms = corner_started_ms_ > 0
        ? std::max<std::int64_t>(
              0, timestamp_ms - corner_started_ms_
          )
        : 0;

    float idle_penalty = 0.0f;
    float corner_penalty = 0.0f;
    if (
        fight.action_allowed &&
        delta_ms > 0 &&
        stationary_ms > config_.idle_grace_ms
    ) {
        idle_penalty =
            config_.idle_penalty_per_second *
            static_cast<float>(delta_ms) / 1000.0f;
    }
    if (
        fight.action_allowed &&
        delta_ms > 0 &&
        corner_ms > config_.corner_grace_ms
    ) {
        corner_penalty =
            config_.corner_penalty_per_second *
            static_cast<float>(delta_ms) / 1000.0f;
    }
    pending_combo_reward_ += combo_reward;
    pending_idle_penalty_ += idle_penalty;
    pending_corner_penalty_ += corner_penalty;
    pending_shaping_reward_ +=
        combo_reward - idle_penalty - corner_penalty;

    const float spatial_confidence = 0.5f * (
        player_tracking_confidence_ +
        opponent_tracking_confidence_
    );
    const float distance =
        std::abs(tracked_player_x_ - tracked_opponent_x_);
    const bool opponent_recent_attack =
        last_damage_taken_ms_ > 0 &&
        timestamp_ms - last_damage_taken_ms_ <=
            config_.attack_memory_ms;
    const bool opponent_motion_attack =
        opponent_track.valid &&
        opponent_track.motion >= 0.18f &&
        distance <= 0.42f;
    const bool player_attacking =
        confirmed_attack ||
        (
            last_player_attack_ms_ > 0 &&
            timestamp_ms - last_player_attack_ms_ <=
                config_.attack_memory_ms
        );
    const bool opponent_attacking =
        opponent_recent_attack || opponent_motion_attack;
    const bool opponent_defending =
        player_attacking &&
        distance <= 0.34f &&
        fight.damage_dealt <= 0.0f &&
        opponent_track.valid &&
        opponent_track.motion < 0.12f;
    const float opponent_action_confidence =
        opponent_recent_attack ? 1.0f :
        opponent_motion_attack ? clamp01(
            0.35f + opponent_track.confidence * 0.45f
        ) :
        opponent_defending ? clamp01(
            0.20f + opponent_track.confidence * 0.35f
        ) : 0.0f;

    latest_ = {
        .frame_id = frame.frame_id,
        .timestamp_ms = timestamp_ms,
        .spatial_valid =
            tracking_initialized_ &&
            spatial_confidence >=
                config_.minimum_spatial_confidence,
        .spatial_confidence = spatial_confidence,
        .opponent_action_confidence =
            opponent_action_confidence,
        .player_x = clamp01(tracked_player_x_),
        .opponent_x = clamp01(tracked_opponent_x_),
        .distance = clamp01(distance),
        .player_on_ground =
            timestamp_ms >= player_airborne_until_ms_ &&
            (
                !player_track.valid ||
                player_bottom_y_ >= 0.74f
            ),
        .opponent_on_ground =
            !opponent_track.valid ||
            opponent_bottom_y_ >= 0.74f,
        .player_defending = confirmed_block,
        .opponent_defending = opponent_defending,
        .player_attacking = player_attacking,
        .opponent_attacking = opponent_attacking,
        .time_since_last_damage_ms =
            last_damage_ms_ > 0
                ? std::clamp(
                      timestamp_ms - last_damage_ms_,
                      std::int64_t{0},
                      config_.time_feature_horizon_ms
                  )
                : config_.time_feature_horizon_ms,
        .consecutive_hits = consecutive_hits_,
        .stationary_time_ms = std::min(
            stationary_ms, config_.time_feature_horizon_ms
        ),
        .corner_time_ms = std::min(
            corner_ms, config_.time_feature_horizon_ms
        ),
        .shaping_reward =
            combo_reward - idle_penalty - corner_penalty,
        .combo_reward = combo_reward,
        .idle_penalty = idle_penalty,
        .corner_penalty = corner_penalty,
    };
    return latest_;
}

CombatStateSnapshot CombatStateTracker::make_snapshot_locked(
    std::int64_t now_ms
) const {
    CombatStateSnapshot result = latest_;
    if (last_damage_ms_ > 0) {
        result.time_since_last_damage_ms = std::clamp(
            now_ms - last_damage_ms_,
            std::int64_t{0},
            config_.time_feature_horizon_ms
        );
    }
    return result;
}

CombatStateSnapshot CombatStateTracker::consume(
    std::int64_t now_ms
) {
    std::lock_guard lock(mutex_);
    CombatStateSnapshot result = make_snapshot_locked(now_ms);
    result.shaping_reward = pending_shaping_reward_;
    result.combo_reward = pending_combo_reward_;
    result.idle_penalty = pending_idle_penalty_;
    result.corner_penalty = pending_corner_penalty_;
    pending_shaping_reward_ = 0.0f;
    pending_combo_reward_ = 0.0f;
    pending_idle_penalty_ = 0.0f;
    pending_corner_penalty_ = 0.0f;
    return result;
}

CombatStateSnapshot CombatStateTracker::snapshot(
    std::int64_t now_ms
) const {
    std::lock_guard lock(mutex_);
    CombatStateSnapshot result = make_snapshot_locked(now_ms);
    result.shaping_reward = 0.0f;
    result.combo_reward = 0.0f;
    result.idle_penalty = 0.0f;
    result.corner_penalty = 0.0f;
    return result;
}

void CombatStateTracker::reset() {
    std::lock_guard lock(mutex_);
    latest_ = {};
    previous_gray_.clear();
    previous_width_ = 0;
    previous_height_ = 0;
    previous_timestamp_ms_ = 0;
    pending_shaping_reward_ = 0.0f;
    pending_combo_reward_ = 0.0f;
    pending_idle_penalty_ = 0.0f;
    pending_corner_penalty_ = 0.0f;
    reset_episode_locked(0);
    latest_.player_x = tracked_player_x_;
    latest_.opponent_x = tracked_opponent_x_;
    latest_.distance =
        std::abs(tracked_player_x_ - tracked_opponent_x_);
    latest_.time_since_last_damage_ms =
        config_.time_feature_horizon_ms;
}

}  // namespace gameagent
