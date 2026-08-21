#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace gameagent {

struct WebSocketServerOptions {
    std::string host = "0.0.0.0";
    std::uint16_t port = 8080;
    std::size_t frame_buffer_capacity = 32;
    std::size_t observation_frames = 8;
    std::size_t applied_action_history_size = 8;
    std::size_t max_jpeg_bytes = 8U * 1024U * 1024U;
    std::size_t persistence_queue_capacity = 256;
    std::chrono::milliseconds decision_interval{50};
    std::chrono::milliseconds failsafe_timeout{200};
    bool timed_sequences_enabled = true;
    std::chrono::milliseconds sequence_input_hold{75};
    std::chrono::milliseconds sequence_neutral_hold{40};
    std::chrono::milliseconds sequence_final_release{100};
    std::size_t sequence_actionable_ticks = 2;
    std::chrono::milliseconds reinforcement_reward_log_interval{5000};
    std::optional<std::string> api_token;
    std::optional<std::uint32_t> random_seed;
    bool default_facing_right = true;
    bool temporal_model_enabled = false;
    bool reinforcement_learning_enabled = false;
    bool reinforcement_training_enabled = true;
    bool reinforcement_resume = true;
    std::size_t reinforcement_checkpoint_interval_episodes = 25;
    std::string reinforcement_backend = "cpp";
    bool reinforcement_trainable_lstm = true;
    bool reinforcement_trainable_resnet = false;
    bool training_reset_state_enabled = true;
    std::chrono::milliseconds training_reset_non_fight_timeout{5000};
    std::chrono::milliseconds training_reset_round_intro_timeout{15000};
    std::chrono::milliseconds reset_ack_timeout{5000};
    std::chrono::milliseconds reset_ready_timeout{10000};
    std::chrono::milliseconds frame_stream_stall_timeout{2000};
    std::chrono::milliseconds frame_stream_recovery_interval{500};
    std::chrono::milliseconds repeated_frame_pause_timeout{2000};
    std::chrono::milliseconds repeated_frame_reset_timeout{3000};
    bool player_is_left = true;
    std::filesystem::path resnet50_checkpoint =
        "/workspace/resnet50_best.ckpt";
    std::filesystem::path temporal_policy_checkpoint;
    std::size_t temporal_visual_projection_size = 256;
    std::size_t temporal_hidden_size = 256;
    int temporal_device_id = 0;
    std::filesystem::path reinforcement_checkpoint_directory =
        "/workspace/gameagent/rl_v21";
    std::filesystem::path recurrent_service_script =
        "/tmp/GameAgent/scripts/recurrent_dqn_service.py";
    std::size_t reinforcement_network_hidden_size = 128;
    std::size_t reinforcement_replay_capacity = 20000;
    std::size_t reinforcement_minimum_replay_size = 256;
    std::size_t reinforcement_batch_size = 64;
    std::uint64_t reinforcement_target_update_interval = 500;
    std::uint64_t reinforcement_epsilon_decay_selections = 100000;
    // Decisions run at 10 Hz. 0.995 keeps a roughly 20-second value horizon
    // and limits long bootstrap chains that previously inflated greedy Q.
    float reinforcement_gamma = 0.995f;
    float reinforcement_q_value_limit = 25.0f;
    float reinforcement_terminal_replay_fraction = 0.25f;
    float reinforcement_hard_negative_replay_fraction = 0.25f;
    float reinforcement_winning_replay_fraction = 0.30f;
    float reinforcement_winning_replay_min_fraction = 0.10f;
    std::size_t reinforcement_winning_replay_ramp_episodes = 5;
    std::size_t reinforcement_winning_replay_capacity = 4096;
    float reinforcement_hard_negative_q_threshold = 1.0f;
    std::size_t reinforcement_n_step_options = 3;
    bool reinforcement_conservative_bootstrap = true;
    float reinforcement_learning_rate = 1.0e-4f;
    float reinforcement_resnet_learning_rate = 1.0e-6f;
    std::size_t reinforcement_resnet_batch_size = 4;
    std::size_t reinforcement_resnet_update_interval = 16;
    std::size_t reinforcement_resnet_replay_images = 4096;
    float reinforcement_damage_dealt_reward_scale = 15.0f;
    float reinforcement_damage_taken_penalty_scale = 11.0f;
    float reinforcement_round_win_bonus = 10.0f;
    float reinforcement_round_loss_penalty = 5.0f;
    float reinforcement_combo_reward_scale = 0.05f;
    float reinforcement_defense_reward_per_second = 0.05f;
    float reinforcement_idle_penalty_per_second = 0.01f;
    float reinforcement_corner_penalty_per_second = 0.02f;
    float reinforcement_neutral_option_penalty = 0.20f;
    bool reinforcement_suppress_neutral_during_fight = true;
    float reinforcement_combo_max_distance = 0.22f;
    float reinforcement_close_attack_max_distance = 0.24f;
    float reinforcement_throw_max_distance = 0.12f;
    float reinforcement_sweep_max_distance = 0.26f;
    float reinforcement_projectile_preferred_min_distance = 0.28f;
    float reinforcement_projectile_q_preference_margin = 1.0f;
    float reinforcement_disable_cloud_min_distance = 0.12f;
    float reinforcement_disable_cloud_max_distance = 0.55f;
    float reinforcement_advance_min_distance = 0.18f;
    float reinforcement_defense_max_distance = 0.32f;
    bool reinforcement_dynamic_facing = true;
    std::size_t reinforcement_disable_cloud_cooldown_options = 2;
    std::size_t reinforcement_throw_cooldown_options = 2;
    std::size_t reinforcement_sweep_cooldown_options = 1;
    std::size_t reinforcement_melee_impact_credit_steps = 2;
    std::size_t reinforcement_teleport_impact_credit_steps = 4;
    std::size_t reinforcement_projectile_impact_credit_min_steps = 4;
    std::size_t reinforcement_projectile_impact_credit_max_steps = 8;
    std::size_t reinforcement_greedy_training_episode_interval = 10;
    float reinforcement_greedy_training_epsilon = 0.05f;
    float reinforcement_epsilon_start = 1.0f;
    float reinforcement_epsilon_final = 0.05f;
    float reinforcement_new_matchup_epsilon_start = 0.35f;
    std::uint64_t reinforcement_new_matchup_epsilon_decay_selections =
        15000;
    float reinforcement_scripted_exploration_probability = 0.50f;
    std::filesystem::path directed_curriculum_path;
    std::filesystem::path metrics_path = "metrics.jsonl";
    std::filesystem::path reinforcement_migration_checkpoint;
    std::filesystem::path capture_directory = "captures";
    std::filesystem::path training_capture_directory =
        "/workspace/gameagent/training_captures";
    bool save_frames = false;
    std::chrono::milliseconds debug_capture_interval{3000};
    bool collect_training_data = false;
};

class WebSocketServer {
public:
    explicit WebSocketServer(WebSocketServerOptions options);

    int run(const std::atomic_bool& stop_requested);

private:
    WebSocketServerOptions options_;
};

}  // namespace gameagent
