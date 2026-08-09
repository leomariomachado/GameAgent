#include "ReinforcementAgent.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<char> read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "could not read checkpoint for immutability test"
        );
    }
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
}

gameagent::ReinforcementDecision complete_one_option(
    gameagent::ReinforcementAgent& agent,
    const std::vector<float>& start,
    const std::vector<float>& next,
    std::uint64_t episode_id,
    std::uint64_t& decision_id
) {
    auto decision = agent.step({
        .state = start,
        .action_allowed = true,
        .episode_started = true,
        .episode_id = episode_id,
        .frame_id = decision_id,
        .decision_id = decision_id,
    });
    ++decision_id;
    require(
        decision.option_steps > 0,
        "shared RL agent did not select an option"
    );
    while (decision.option_step < decision.option_steps) {
        decision = agent.step({
            .state = start,
            .reward = 0.05f,
            .action_allowed = true,
            .episode_id = episode_id,
            .frame_id = decision_id,
            .decision_id = decision_id,
        });
        ++decision_id;
        require(decision_id < 200, "RL option did not terminate");
    }
    return agent.step({
        .state = next,
        .reward = 0.5f,
        .action_allowed = true,
        .episode_id = episode_id,
        .frame_id = decision_id,
        .decision_id = decision_id,
    });
}

}  // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        "gameagent_shared_reinforcement_test";
    const std::filesystem::path latest =
        directory / "shared_policy_latest.garl";
    const std::filesystem::path subzero_best =
        directory / "shared_policy_best_subzero.garl";
    const std::filesystem::path scorpion_best =
        directory / "shared_policy_best_scorpion.garl";
    try {
        std::filesystem::remove_all(directory);
        gameagent::ReinforcementAgentConfig config{
            .input_size = 8,
            .network_hidden_size = 16,
            .replay_capacity = 64,
            .minimum_replay_size = 1,
            .batch_size = 1,
            .training_updates_per_transition = 1,
            .target_update_interval = 2,
            .gamma = 0.95f,
            .learning_rate = 1.0e-3f,
            .epsilon_start = 0.0f,
            .epsilon_final = 0.0f,
            .epsilon_decay_selections = 1,
            .new_matchup_epsilon_start = 0.0f,
            .new_matchup_epsilon_decay_selections = 1,
            .scripted_exploration_probability = 0.5f,
            .neutral_option_penalty = 0.0f,
            .random_seed = 12345,
            .character = gameagent::FighterCharacter::subzero,
            .opponent_character =
                gameagent::OpponentCharacter::scorpion,
            .character_embedding_size = 4,
            .opponent_embedding_size = 4,
            .training_enabled = true,
            .resume = false,
            .save_replay = true,
            .checkpoint_path = latest,
            .best_checkpoint_path = subzero_best,
        };
        const std::vector<float> state_a{
            0.1f, -0.2f, 0.3f, -0.4f,
            0.5f, 0.6f, 0.7f, 0.8f
        };
        const std::vector<float> state_b{
            0.2f, -0.1f, 0.4f, -0.3f,
            0.6f, 0.5f, 0.8f, 0.7f
        };

        auto suspended_config = config;
        suspended_config.checkpoint_path =
            directory / "suspended_latest.garl";
        suspended_config.best_checkpoint_path =
            directory / "suspended_best.garl";
        gameagent::ReinforcementAgent suspended(suspended_config);
        auto suspended_decision = suspended.step({
            .state = state_a,
            .action_allowed = true,
            .episode_started = true,
            .episode_id = 1,
            .frame_id = 1,
            .decision_id = 1,
        });
        require(
            suspended_decision.option_steps > 0,
            "suspension test did not start an option"
        );
        suspended.suspend();
        const auto before_suspended_ticks = suspended.stats();
        for (std::uint64_t tick = 2; tick < 32; ++tick) {
            suspended_decision = suspended.step({
                .reward = 0.0f,
                .action_allowed = false,
                .episode_id = 1,
                .frame_id = tick,
                .decision_id = tick,
            });
            require(
                suspended_decision.commands.empty(),
                "suspended agent emitted a command"
            );
        }
        const auto after_suspended_ticks = suspended.stats();
        require(
            after_suspended_ticks.training_updates ==
                    before_suspended_ticks.training_updates &&
                after_suspended_ticks.replay_size ==
                    before_suspended_ticks.replay_size &&
                after_suspended_ticks.action_selections ==
                    before_suspended_ticks.action_selections &&
                after_suspended_ticks.epsilon ==
                    before_suspended_ticks.epsilon,
            "suspended ticks changed replay, optimizer or epsilon"
        );
        suspended_decision = suspended.step({
            .state = state_b,
            .action_allowed = true,
            .episode_id = 1,
            .frame_id = 32,
            .decision_id = 32,
        });
        require(
            suspended_decision.option_steps > 0 &&
                suspended.stats().action_selections ==
                    before_suspended_ticks.action_selections + 1,
            "agent did not resume cleanly after suspension"
        );

        gameagent::ReinforcementAgent subzero(config);
        require(
            subzero.option_names().size() == 27,
            "Sub-Zero did not receive its action head"
        );
        const double initial_shared_checksum =
            subzero.stats().shared_parameter_checksum;
        std::uint64_t decision_id = 1;
        auto decision = complete_one_option(
            subzero, state_a, state_b, 1, decision_id
        );
        require(
            decision.training_performed &&
                decision.replay_size >= 1 &&
                std::isfinite(decision.training_loss),
            "Sub-Zero experience did not train the shared trunk"
        );
        decision = subzero.step({
            .state = state_b,
            .reward = 1.0f,
            .action_allowed = false,
            .episode_ended = true,
            .terminal = true,
            .episode_id = 1,
            .frame_id = decision_id + 1,
            .decision_id = decision_id + 1,
        });
        require(
            decision.commands.empty() && decision.checkpoint_saved,
            "terminal state did not release controls and save shared policy"
        );
        const auto subzero_stats = subzero.stats();
        require(
            subzero_stats.shared_parameter_checksum !=
                initial_shared_checksum &&
                subzero_stats.shared_replay_matchups == 1,
            "shared parameters/replay were not updated"
        );
        require(
            std::filesystem::is_regular_file(latest) &&
                std::filesystem::is_regular_file(subzero_best),
            "Sub-Zero did not write shared latest/best checkpoints"
        );

        const auto before_reconnect = subzero.stats();
        subzero.configure_session(
            gameagent::FighterCharacter::subzero,
            gameagent::OpponentCharacter::scorpion,
            gameagent::FighterFacing::right,
            subzero_best
        );
        const auto after_reconnect = subzero.stats();
        require(
            after_reconnect.replay_size == before_reconnect.replay_size &&
                after_reconnect.training_updates ==
                    before_reconnect.training_updates &&
                after_reconnect.action_selections ==
                    before_reconnect.action_selections &&
                after_reconnect.matchup_action_selections ==
                    before_reconnect.matchup_action_selections &&
                after_reconnect.current_episode_return == 0.0f,
            "WebSocket session reconfiguration discarded replay or learner state"
        );
        subzero.configure_session(
            gameagent::FighterCharacter::subzero,
            gameagent::OpponentCharacter::kitana,
            gameagent::FighterFacing::right,
            subzero_best
        );
        const auto unseen_matchup = subzero.stats();
        require(
            unseen_matchup.action_selections ==
                    before_reconnect.action_selections &&
                unseen_matchup.matchup_action_selections == 0 &&
                unseen_matchup.matchup_episodes_completed == 0 &&
                unseen_matchup.matchup_wins == 0,
            "new opponent did not receive independent exploration progress"
        );
        subzero.configure_session(
            gameagent::FighterCharacter::subzero,
            gameagent::OpponentCharacter::scorpion,
            gameagent::FighterFacing::right,
            subzero_best
        );
        require(
            subzero.stats().matchup_action_selections ==
                before_reconnect.matchup_action_selections,
            "returning matchup did not restore its exploration progress"
        );

        auto scorpion_config = config;
        scorpion_config.character =
            gameagent::FighterCharacter::scorpion;
        scorpion_config.opponent_character =
            gameagent::OpponentCharacter::sindel;
        scorpion_config.facing = gameagent::FighterFacing::left;
        scorpion_config.resume = true;
        scorpion_config.best_checkpoint_path = scorpion_best;
        gameagent::ReinforcementAgent scorpion(scorpion_config);
        const auto inherited = scorpion.stats();
        require(
            inherited.checkpoint_loaded &&
                inherited.training_updates ==
                    subzero_stats.training_updates &&
                inherited.replay_size == subzero_stats.replay_size &&
                inherited.shared_parameter_checksum ==
                    subzero_stats.shared_parameter_checksum,
            "Scorpion did not inherit Sub-Zero shared knowledge"
        );
        require(
            inherited.action_selections == 0 &&
                scorpion.option_names().size() == 29,
            "Scorpion exploration/head was not kept character-specific"
        );

        decision_id = 100;
        decision = complete_one_option(
            scorpion, state_b, state_a, 2, decision_id
        );
        require(
            decision.training_performed,
            "Scorpion experience did not train shared policy"
        );
        decision = scorpion.step({
            .state = state_a,
            .reward = 0.75f,
            .action_allowed = false,
            .episode_ended = true,
            .terminal = true,
            .episode_id = 2,
            .frame_id = decision_id + 1,
            .decision_id = decision_id + 1,
        });
        const auto scorpion_stats = scorpion.stats();
        require(
            scorpion_stats.shared_replay_matchups == 2 &&
                scorpion_stats.training_updates >
                    subzero_stats.training_updates &&
                std::filesystem::is_regular_file(scorpion_best),
            "shared replay did not retain both matchups"
        );

        config.resume = true;
        gameagent::ReinforcementAgent resumed_subzero(config);
        const auto transferred_back = resumed_subzero.stats();
        require(
            transferred_back.checkpoint_loaded &&
                transferred_back.training_updates ==
                    scorpion_stats.training_updates &&
                transferred_back.replay_size ==
                    scorpion_stats.replay_size &&
                transferred_back.shared_replay_matchups == 2,
            "Sub-Zero did not receive Scorpion shared updates"
        );
        require(
            transferred_back.action_selections ==
                subzero_stats.action_selections,
            "per-character epsilon progress was not preserved"
        );

        const std::vector<
            std::pair<gameagent::FighterCharacter, std::size_t>
        > heads{
            {gameagent::FighterCharacter::subzero, 27U},
            {gameagent::FighterCharacter::noob_saibot, 25U},
            {gameagent::FighterCharacter::scorpion, 29U},
            {gameagent::FighterCharacter::sindel, 30U},
        };
        for (const auto& [character, expected_options] : heads) {
            auto head_config = config;
            head_config.character = character;
            head_config.opponent_character =
                gameagent::OpponentCharacter::shao_kahn;
            gameagent::ReinforcementAgent character_agent(head_config);
            require(
                character_agent.stats().checkpoint_loaded &&
                    character_agent.option_names().size() ==
                        expected_options,
                "shared checkpoint did not expose the requested action head"
            );
        }

        const std::vector<char> checkpoint_before = read_file(latest);
        const std::vector<char> best_before = read_file(subzero_best);
        auto evaluation_config = config;
        evaluation_config.training_enabled = false;
        evaluation_config.checkpoint_writes_enabled = false;
        evaluation_config.epsilon_start = 0.0f;
        evaluation_config.epsilon_final = 0.0f;
        evaluation_config.new_matchup_epsilon_start = 0.35f;
        evaluation_config.character =
            gameagent::FighterCharacter::subzero;
        evaluation_config.opponent_character =
            gameagent::OpponentCharacter::scorpion;
        gameagent::ReinforcementAgent evaluation(evaluation_config);
        const auto evaluation_before = evaluation.stats();
        decision_id = 20;
        decision = complete_one_option(
            evaluation, state_a, state_b, 20, decision_id
        );
        require(
            decision.selection_source == "dqn_exploit" &&
                decision.epsilon == 0.0f &&
                !decision.training_performed,
            "read-only evaluation did not use pure DQN inference"
        );
        decision = evaluation.step({
            .state = state_b,
            .reward = -1.0f,
            .action_allowed = false,
            .episode_ended = true,
            .terminal = true,
            .episode_id = 20,
            .frame_id = decision_id + 1,
            .decision_id = decision_id + 1,
        });
        evaluation.reset_episode(true);
        evaluation.save_checkpoint(
            directory / "evaluation_must_not_write.garl"
        );
        const auto evaluation_after = evaluation.stats();
        require(
            !decision.checkpoint_saved &&
                evaluation_after.training_updates ==
                    evaluation_before.training_updates &&
                evaluation_after.replay_size ==
                    evaluation_before.replay_size &&
                evaluation_after.action_selections ==
                    evaluation_before.action_selections &&
                evaluation_after.matchup_action_selections ==
                    evaluation_before.matchup_action_selections &&
                evaluation_after.episodes_completed ==
                    evaluation_before.episodes_completed &&
                evaluation_after.matchup_episodes_completed ==
                    evaluation_before.matchup_episodes_completed &&
                evaluation_after.matchup_wins ==
                    evaluation_before.matchup_wins &&
                evaluation_after.shared_parameter_checksum ==
                    evaluation_before.shared_parameter_checksum,
            "read-only evaluation changed learner or scheduler state"
        );
        require(
            read_file(latest) == checkpoint_before &&
                read_file(subzero_best) == best_before &&
                !std::filesystem::exists(
                    directory / "evaluation_must_not_write.garl"
                ),
            "read-only evaluation modified a checkpoint"
        );

        const auto interval_latest =
            directory / "interval_latest.garl";
        const auto interval_best =
            directory / "interval_best.garl";
        auto interval_config = config;
        interval_config.resume = false;
        interval_config.checkpoint_interval_episodes = 2;
        interval_config.checkpoint_path = interval_latest;
        interval_config.best_checkpoint_path = interval_best;
        gameagent::ReinforcementAgent interval_agent(interval_config);
        auto interval_decision = interval_agent.step({
            .state = state_a,
            .reward = -1.0f,
            .action_allowed = false,
            .episode_started = true,
            .episode_ended = true,
            .terminal = true,
            .episode_id = 30,
            .frame_id = 30,
            .decision_id = 30,
        });
        require(
            !interval_decision.checkpoint_saved &&
                !std::filesystem::exists(interval_latest),
            "periodic checkpoint was written before its interval"
        );
        interval_decision = interval_agent.step({
            .state = state_a,
            .reward = -2.0f,
            .action_allowed = false,
            .episode_started = true,
            .episode_ended = true,
            .terminal = true,
            .episode_id = 31,
            .frame_id = 31,
            .decision_id = 31,
        });
        require(
            interval_decision.checkpoint_saved &&
                std::filesystem::is_regular_file(interval_latest),
            "periodic checkpoint was not written at its interval"
        );

        auto neutral_config = config;
        neutral_config.resume = false;
        // Tactical sampling tests intentionally exercise epsilon=1. Runtime
        // V2.1 evaluation is always epsilon=0 and is covered separately.
        neutral_config.training_enabled = true;
        neutral_config.checkpoint_writes_enabled = false;
        neutral_config.epsilon_start = 1.0f;
        neutral_config.epsilon_final = 1.0f;
        neutral_config.new_matchup_epsilon_start = 1.0f;
        neutral_config.scripted_exploration_probability = 0.0f;
        neutral_config.neutral_option_penalty = 0.25f;
        neutral_config.random_seed = 9917;
        gameagent::ReinforcementAgent neutral_agent(neutral_config);
        neutral_agent.reset_episode(false);
        bool observed_neutral = false;
        decision_id = 1000;
        for (std::size_t attempt = 0;
             attempt < 256 && !observed_neutral;
             ++attempt) {
            const auto neutral_decision = neutral_agent.step({
                .state = state_a,
                .action_allowed = true,
                .episode_started = attempt == 0,
                .episode_id = 40,
                .frame_id = decision_id,
                .decision_id = decision_id,
            });
            ++decision_id;
            if (neutral_decision.option_name == "Neutro") {
                observed_neutral = true;
                require(
                    std::abs(neutral_decision.neutral_penalty - 0.25f) <
                            1.0e-6f &&
                        std::abs(neutral_decision.reward_consumed + 0.25f) <
                            1.0e-6f &&
                        std::abs(neutral_decision.episode_return + 0.25f) <
                            1.0e-6f,
                    "neutral option did not receive its configured cost"
                );
            }
        }
        require(
            observed_neutral,
            "random exploration never selected the neutral option"
        );

        auto cooldown_config = neutral_config;
        cooldown_config.character =
            gameagent::FighterCharacter::noob_saibot;
        cooldown_config.scripted_exploration_probability = 1.0f;
        cooldown_config.neutral_option_penalty = 0.0f;
        cooldown_config.melee_impact_credit_steps = 2U;
        cooldown_config.teleport_impact_credit_steps = 2U;
        cooldown_config.projectile_impact_credit_min_steps = 2U;
        cooldown_config.projectile_impact_credit_max_steps = 2U;
        cooldown_config.random_seed = 88117;
        gameagent::ReinforcementAgent cooldown_agent(cooldown_config);
        std::size_t selected_options = 0;
        std::size_t cloud_options = 0;
        std::size_t teleport_options = 0;
        std::size_t throw_options = 0;
        std::size_t sweep_options = 0;
        std::optional<std::size_t> last_cloud_option;
        std::optional<std::size_t> last_teleport_option;
        std::optional<std::size_t> last_throw_option;
        std::optional<std::size_t> last_sweep_option;
        bool observed_unrestricted_teleport = false;
        bool observed_delayed_credit = false;
        decision_id = 2000;
        for (std::size_t tick = 0;
             tick < 10000 && selected_options < 600;
             ++tick) {
            const auto cooldown_decision = cooldown_agent.step({
                .state = state_a,
                .action_allowed = true,
                .episode_started = tick == 0,
                .episode_id = 50,
                .frame_id = decision_id,
                .decision_id = decision_id,
            });
            ++decision_id;
            if (cooldown_decision.impact_credit_step > 0U) {
                observed_delayed_credit = true;
                require(
                    cooldown_decision.commands.empty() &&
                        cooldown_decision.impact_credit_steps == 2U,
                    "delayed impact credit did not hold a released state"
                );
            }
            if (cooldown_decision.option_step != 1U) {
                continue;
            }
            const bool is_cloud =
                cooldown_decision.option_name.rfind(
                    "Nuvem de Desativação", 0
                ) == 0;
            if (is_cloud) {
                if (last_cloud_option.has_value()) {
                    require(
                        selected_options >=
                            *last_cloud_option + 3U,
                        "DQN policy selected a disable cloud during cooldown"
                    );
                }
                last_cloud_option = selected_options;
                ++cloud_options;
            }
            const bool is_teleport =
                cooldown_decision.option_name ==
                "Teleporte com Arremesso";
            if (is_teleport) {
                if (last_teleport_option.has_value()) {
                    observed_unrestricted_teleport =
                        observed_unrestricted_teleport ||
                        selected_options < *last_teleport_option + 4U;
                }
                require(
                    cooldown_decision.option_steps == 6U,
                    "teleport did not expose its delayed credit window"
                );
                last_teleport_option = selected_options;
                ++teleport_options;
            }
            const bool is_throw =
                cooldown_decision.option_name == "Agarrão" ||
                cooldown_decision.option_name.rfind(
                    "Agarrão [Espelho", 0
                ) == 0;
            if (is_throw) {
                if (last_throw_option.has_value()) {
                    require(
                        selected_options >= *last_throw_option + 3U,
                        "DQN selected another throw during cooldown"
                    );
                }
                last_throw_option = selected_options;
                ++throw_options;
            }
            const bool is_sweep =
                cooldown_decision.option_name.rfind("Rasteira", 0) == 0;
            if (is_sweep) {
                if (last_sweep_option.has_value()) {
                    require(
                        selected_options >= *last_sweep_option + 2U,
                        "DQN selected another sweep during cooldown"
                    );
                }
                last_sweep_option = selected_options;
                ++sweep_options;
            }
            ++selected_options;
        }
        require(
            selected_options == 600 && cloud_options > 10 &&
                teleport_options > 10 && observed_unrestricted_teleport &&
                throw_options > 10 && sweep_options > 10 &&
                observed_delayed_credit,
            "DQN cooldown/credit test did not sample enough options"
        );

        auto grounded_mask_config = cooldown_config;
        grounded_mask_config.random_seed = 88217;
        gameagent::ReinforcementAgent grounded_mask_agent(
            grounded_mask_config
        );
        selected_options = 0;
        decision_id = 12000;
        while (selected_options < 200U && decision_id < 24000U) {
            const auto masked = grounded_mask_agent.step({
                .state = state_a,
                .action_allowed = true,
                .episode_started = decision_id == 12000U,
                .episode_id = 51,
                .frame_id = decision_id,
                .decision_id = decision_id,
                .fighter_distance = 0.08f,
                .player_x = 0.20f,
                .opponent_x = 0.28f,
                .spatial_valid = true,
                .player_on_ground = false,
                .opponent_on_ground = true,
            });
            ++decision_id;
            if (masked.option_step != 1U) {
                continue;
            }
            require(
                masked.option_name != "Agarrão" &&
                    masked.option_name.rfind(
                        "Agarrão [Espelho", 0
                    ) != 0 &&
                    masked.option_name.rfind("Rasteira", 0) != 0,
                "ground-only action was allowed while the player was airborne"
            );
            ++selected_options;
        }
        require(
            selected_options == 200U,
            "ground-only tactical mask did not complete its sample"
        );

        auto facing_config = cooldown_config;
        facing_config.random_seed = 99117;
        gameagent::ReinforcementAgent facing_right_agent(facing_config);
        gameagent::ReinforcementAgent facing_left_agent(facing_config);
        bool observed_dynamic_mirror = false;
        for (std::size_t tick = 0; tick < 2000U; ++tick) {
            const auto right = facing_right_agent.step({
                .state = state_a,
                .action_allowed = true,
                .episode_started = tick == 0,
                .episode_id = 55,
                .frame_id = 10000U + tick,
                .decision_id = 10000U + tick,
                .fighter_distance = 0.40f,
                .player_x = 0.20f,
                .opponent_x = 0.70f,
                .spatial_valid = true,
            });
            const auto left = facing_left_agent.step({
                .state = state_a,
                .action_allowed = true,
                .episode_started = tick == 0,
                .episode_id = 55,
                .frame_id = 12000U + tick,
                .decision_id = 12000U + tick,
                .fighter_distance = 0.40f,
                .player_x = 0.70f,
                .opponent_x = 0.20f,
                .spatial_valid = true,
            });
            require(
                right.option_name == left.option_name &&
                    right.commands.size() == left.commands.size(),
                "dynamic-facing peers diverged before command mirroring"
            );
            for (std::size_t index = 0;
                 index < right.commands.size(); ++index) {
                const int a = right.commands[index];
                const int b = left.commands[index];
                if (a == 8 || a == 9) {
                    require(
                        (a == 8 && b == 9) || (a == 9 && b == 8),
                        "dynamic facing did not swap horizontal input"
                    );
                    observed_dynamic_mirror = true;
                } else {
                    require(
                        a == b,
                        "dynamic facing changed a non-horizontal input"
                    );
                }
            }
            if (observed_dynamic_mirror) {
                require(
                    !right.commands_mirrored && left.commands_mirrored,
                    "dynamic-facing metadata is inconsistent"
                );
                break;
            }
        }
        require(
            observed_dynamic_mirror,
            "dynamic-facing test never emitted a directional command"
        );

        auto far_config = cooldown_config;
        far_config.random_seed = 77231;
        gameagent::ReinforcementAgent far_agent(far_config);
        selected_options = 0;
        decision_id = 20000;
        while (selected_options < 200U && decision_id < 40000U) {
            const auto far_decision = far_agent.step({
                .state = state_a,
                .action_allowed = true,
                .episode_started = decision_id == 20000U,
                .episode_id = 60,
                .frame_id = decision_id,
                .decision_id = decision_id,
                .fighter_distance = 0.70f,
                .spatial_valid = true,
            });
            ++decision_id;
            if (far_decision.option_step != 1U) {
                continue;
            }
            require(
                far_decision.option_name.rfind(
                    "Arremesso de Sombra", 0
                ) == 0 ||
                    far_decision.option_name.rfind(
                        "Nuvem de Desativação", 0
                    ) == 0,
                "far scripted exploration did not prefer a projectile"
            );
            require(
                far_decision.option_name.find("[Espelho F/B]") ==
                    std::string::npos,
                "dynamic facing allowed an explicit mirrored duplicate"
            );
            require(
                far_decision.option_name.find("Golpes") ==
                    std::string::npos,
                "far tactical mask allowed a close combo"
            );
            ++selected_options;
        }
        require(
            selected_options == 200U,
            "far tactical policy did not complete its sample"
        );

        auto near_config = cooldown_config;
        near_config.random_seed = 88119;
        gameagent::ReinforcementAgent near_agent(near_config);
        bool observed_near_combo = false;
        decision_id = 40000;
        for (std::size_t tick = 0;
             tick < 10000U && !observed_near_combo;
             ++tick) {
            const auto near_decision = near_agent.step({
                .state = state_a,
                .action_allowed = true,
                .episode_started = tick == 0,
                .episode_id = 61,
                .frame_id = decision_id,
                .decision_id = decision_id,
                .fighter_distance = 0.10f,
                .spatial_valid = true,
            });
            ++decision_id;
            observed_near_combo =
                near_decision.option_step == 1U &&
                near_decision.option_name.find("Golpes") !=
                    std::string::npos;
            if (near_decision.option_step == 1U) {
                require(
                    near_decision.option_name.rfind(
                        "Arremesso de Sombra", 0
                    ) != 0 &&
                        near_decision.option_name.rfind(
                            "Nuvem de Desativação", 0
                        ) != 0,
                    "near tactical mask allowed a projectile"
                );
            }
        }
        require(
            observed_near_combo,
            "near tactical mask never released a close combo"
        );

        auto active_primitive_config = neutral_config;
        active_primitive_config.random_seed = 55119;
        gameagent::ReinforcementAgent active_primitive_agent(
            active_primitive_config
        );
        selected_options = 0;
        decision_id = 50000;
        while (selected_options < 200U && decision_id < 60000U) {
            const auto active_decision = active_primitive_agent.step({
                .state = state_a,
                .action_allowed = true,
                .episode_started = decision_id == 50000U,
                .episode_id = 62,
                .frame_id = decision_id,
                .decision_id = decision_id,
                .fighter_distance = 0.40f,
                .spatial_valid = true,
            });
            ++decision_id;
            if (active_decision.option_step != 1U) {
                continue;
            }
            require(
                active_decision.option_name != "Neutro",
                "active-fight tactical mask selected Neutral"
            );
            require(
                active_decision.allowed_options > 0U,
                "active-fight tactical mask reported no options"
            );
            ++selected_options;
        }
        require(
            selected_options == 200U,
            "active-fight neutral suppression did not complete its sample"
        );

        std::cout
            << "[INFO] shared_updates="
            << transferred_back.training_updates
            << " shared_replay="
            << transferred_back.replay_size
            << " matchups="
            << transferred_back.shared_replay_matchups
            << " subzero_epsilon="
            << transferred_back.epsilon
            << " scorpion_epsilon="
            << scorpion_stats.epsilon
            << " evaluation_epsilon="
            << evaluation_after.epsilon
            << " checkpoint_unchanged=true\n"
            << "[PASS] shared multi-character Double DQN test passed\n";
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(directory);
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
