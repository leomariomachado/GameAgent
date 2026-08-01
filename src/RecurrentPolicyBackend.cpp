#include "RecurrentPolicyBackend.h"
#include "StreamProtocol.h"

#include <json.hpp>

#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

extern char** environ;

namespace gameagent {
namespace {

std::string join_sizes(std::span<const std::size_t> values) {
    std::string result;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            result.push_back(',');
        }
        result += std::to_string(values[index]);
    }
    return result;
}

void require_positive(std::size_t value, const char* name) {
    if (value == 0) {
        throw std::invalid_argument(
            std::string("recurrent backend ") + name + " must be positive"
        );
    }
}

}  // namespace

struct RecurrentPolicyBackend::Implementation {
    explicit Implementation(RecurrentPolicyBackendConfig value)
        : config(std::move(value)) {
        validate();
        spawn_service();
        const nlohmann::json ready = read_response();
        update_stats(ready);
        if (!ready.value("ready", false)) {
            throw std::runtime_error(
                "recurrent policy service did not report ready"
            );
        }
    }

    ~Implementation() {
        shutdown();
    }

    void validate() const {
        if (
            config.service_script.empty() ||
            !std::filesystem::is_regular_file(config.service_script)
        ) {
            throw std::invalid_argument(
                "recurrent policy service script was not found: " +
                config.service_script.string()
            );
        }
        if (config.checkpoint_path.empty() || config.device_id < 0) {
            throw std::invalid_argument(
                "recurrent policy checkpoint/device is invalid"
            );
        }
        require_positive(config.visual_size, "visual size");
        require_positive(config.projection_size, "projection size");
        require_positive(config.controller_size, "controller size");
        require_positive(config.temporal_hidden_size, "hidden size");
        require_positive(config.context_size, "context size");
        require_positive(config.sequence_capacity, "sequence capacity");
        require_positive(config.replay_capacity, "replay capacity");
        require_positive(config.minimum_replay_size, "minimum replay");
        require_positive(config.batch_size, "batch size");
        require_positive(config.n_step_options, "n-step options");
        if (config.trainable_resnet) {
            if (
                config.resnet50_checkpoint_path.empty() ||
                !std::filesystem::is_regular_file(
                    config.resnet50_checkpoint_path
                )
            ) {
                throw std::invalid_argument(
                    "trainable ResNet50 checkpoint was not found: " +
                    config.resnet50_checkpoint_path.string()
                );
            }
            require_positive(
                config.resnet_batch_size, "ResNet batch size"
            );
            require_positive(
                config.resnet_update_interval,
                "ResNet update interval"
            );
            require_positive(
                config.resnet_replay_images,
                "ResNet replay images"
            );
        }
        require_positive(
            config.training_updates_per_transition,
            "updates per transition"
        );
        require_positive(
            static_cast<std::size_t>(config.target_update_interval),
            "target update interval"
        );
        if (
            config.action_sizes.empty() ||
            config.minimum_replay_size > config.replay_capacity ||
            config.batch_size > config.replay_capacity ||
            config.gamma <= 0.0f || config.gamma > 1.0f ||
            config.q_value_limit <= 0.0f ||
            config.terminal_replay_fraction < 0.0f ||
            config.terminal_replay_fraction > 1.0f ||
            config.hard_negative_replay_fraction < 0.0f ||
            config.hard_negative_replay_fraction > 1.0f ||
            config.terminal_replay_fraction +
                    config.hard_negative_replay_fraction >
                1.0f ||
            config.hard_negative_q_threshold < 0.0f ||
            config.learning_rate <= 0.0f ||
            config.resnet_learning_rate <= 0.0f ||
            config.gradient_clip <= 0.0f
        ) {
            throw std::invalid_argument(
                "invalid trainable recurrent policy configuration"
            );
        }
        for (std::size_t size : config.action_sizes) {
            require_positive(size, "action head size");
        }
    }

    void spawn_service() {
        int child_stdin[2]{-1, -1};
        int child_stdout[2]{-1, -1};
        if (pipe(child_stdin) != 0 || pipe(child_stdout) != 0) {
            const int error = errno;
            if (child_stdin[0] >= 0) {
                close(child_stdin[0]);
                close(child_stdin[1]);
            }
            if (child_stdout[0] >= 0) {
                close(child_stdout[0]);
                close(child_stdout[1]);
            }
            throw std::runtime_error(
                "could not create recurrent service pipes: " +
                std::string(std::strerror(error))
            );
        }

        std::vector<std::string> arguments{
            "python3",
            "-u",
            config.service_script.string(),
            "--checkpoint", config.checkpoint_path.string(),
            "--migration-checkpoint",
            config.migration_checkpoint_path.string(),
            "--device-id", std::to_string(config.device_id),
            "--training", config.training_enabled ? "1" : "0",
            "--checkpoint-writes",
            config.checkpoint_writes_enabled ? "1" : "0",
            "--resume", config.resume ? "1" : "0",
            "--visual-size", std::to_string(config.visual_size),
            "--projection-size", std::to_string(config.projection_size),
            "--controller-size", std::to_string(config.controller_size),
            "--temporal-hidden-size",
            std::to_string(config.temporal_hidden_size),
            "--context-size", std::to_string(config.context_size),
            "--strategy-hidden-size",
            std::to_string(config.strategy_hidden_size),
            "--sequence-capacity",
            std::to_string(config.sequence_capacity),
            "--character-embedding-size",
            std::to_string(config.character_embedding_size),
            "--opponent-embedding-size",
            std::to_string(config.opponent_embedding_size),
            "--opponent-count", std::to_string(config.opponent_count),
            "--action-sizes", join_sizes(config.action_sizes),
            "--replay-capacity", std::to_string(config.replay_capacity),
            "--minimum-replay-size",
            std::to_string(config.minimum_replay_size),
            "--batch-size", std::to_string(config.batch_size),
            "--target-update",
            std::to_string(config.target_update_interval),
            "--updates-per-transition",
            std::to_string(config.training_updates_per_transition),
            "--initial-updates",
            std::to_string(config.initial_training_updates),
            "--gamma", std::to_string(config.gamma),
            "--conservative-bootstrap",
            config.conservative_bootstrap ? "1" : "0",
            "--q-value-limit", std::to_string(config.q_value_limit),
            "--terminal-replay-fraction",
            std::to_string(config.terminal_replay_fraction),
            "--hard-negative-replay-fraction",
            std::to_string(config.hard_negative_replay_fraction),
            "--hard-negative-q-threshold",
            std::to_string(config.hard_negative_q_threshold),
            "--n-step-options", std::to_string(config.n_step_options),
            "--learning-rate", std::to_string(config.learning_rate),
            "--trainable-resnet",
            config.trainable_resnet ? "1" : "0",
            "--resnet-checkpoint",
            config.resnet50_checkpoint_path.string(),
            "--resnet-learning-rate",
            std::to_string(config.resnet_learning_rate),
            "--resnet-batch-size",
            std::to_string(config.resnet_batch_size),
            "--resnet-update-interval",
            std::to_string(config.resnet_update_interval),
            "--resnet-replay-images",
            std::to_string(config.resnet_replay_images),
            "--gradient-clip", std::to_string(config.gradient_clip),
            "--seed", std::to_string(config.random_seed),
        };
        std::vector<char*> argv;
        argv.reserve(arguments.size() + 1U);
        for (std::string& argument : arguments) {
            argv.push_back(argument.data());
        }
        argv.push_back(nullptr);

        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(
            &actions, child_stdin[0], STDIN_FILENO
        );
        posix_spawn_file_actions_adddup2(
            &actions, child_stdout[1], STDOUT_FILENO
        );
        posix_spawn_file_actions_addclose(&actions, child_stdin[1]);
        posix_spawn_file_actions_addclose(&actions, child_stdout[0]);
        posix_spawn_file_actions_addclose(&actions, child_stdin[0]);
        posix_spawn_file_actions_addclose(&actions, child_stdout[1]);
        const int status = posix_spawnp(
            &child_pid,
            "python3",
            &actions,
            nullptr,
            argv.data(),
            environ
        );
        posix_spawn_file_actions_destroy(&actions);
        close(child_stdin[0]);
        close(child_stdout[1]);
        if (status != 0) {
            close(child_stdin[1]);
            close(child_stdout[0]);
            child_pid = -1;
            throw std::runtime_error(
                "could not launch recurrent policy service: " +
                std::string(std::strerror(status))
            );
        }
        service_input = fdopen(child_stdin[1], "w");
        service_output = fdopen(child_stdout[0], "r");
        if (service_input == nullptr || service_output == nullptr) {
            shutdown();
            throw std::runtime_error(
                "could not open recurrent policy service streams"
            );
        }
        setvbuf(service_input, nullptr, _IOLBF, 0);
    }

    nlohmann::json read_response() {
        char* line = nullptr;
        std::size_t capacity = 0;
        const ssize_t length = getline(
            &line, &capacity, service_output
        );
        if (length < 0) {
            std::free(line);
            throw std::runtime_error(
                "recurrent policy service closed its output"
            );
        }
        const std::string encoded(line, static_cast<std::size_t>(length));
        std::free(line);
        nlohmann::json response = nlohmann::json::parse(encoded);
        if (!response.value("ok", false)) {
            throw std::runtime_error(
                "recurrent policy service failed: " +
                response.value("error", "unknown error")
            );
        }
        return response;
    }

    nlohmann::json transact(const nlohmann::json& request) {
        if (service_input == nullptr || service_output == nullptr) {
            throw std::runtime_error(
                "recurrent policy service is unavailable"
            );
        }
        const std::string encoded = request.dump();
        if (
            std::fwrite(
                encoded.data(), 1, encoded.size(), service_input
            ) != encoded.size() ||
            std::fputc('\n', service_input) == EOF ||
            std::fflush(service_input) != 0
        ) {
            throw std::runtime_error(
                "could not write to recurrent policy service"
            );
        }
        nlohmann::json response = read_response();
        update_stats(response);
        return response;
    }

    void update_stats(const nlohmann::json& response) {
        stats.training_updates =
            response.value("training_updates", stats.training_updates);
        stats.replay_size =
            response.value("replay_size", stats.replay_size);
        stats.matchup_count =
            response.value("matchup_count", stats.matchup_count);
        stats.sequence_size =
            response.value("sequence_size", stats.sequence_size);
        stats.last_loss =
            response.value("last_loss", stats.last_loss);
        stats.last_predicted_q_mean = response.value(
            "last_predicted_q_mean", stats.last_predicted_q_mean
        );
        stats.last_td_target_mean = response.value(
            "last_td_target_mean", stats.last_td_target_mean
        );
        stats.last_bootstrap_mean = response.value(
            "last_bootstrap_mean", stats.last_bootstrap_mean
        );
        stats.last_q_target_gap = response.value(
            "last_q_target_gap", stats.last_q_target_gap
        );
        stats.last_abs_td_error = response.value(
            "last_abs_td_error", stats.last_abs_td_error
        );
        stats.last_q_target_clipped_fraction = response.value(
            "last_q_target_clipped_fraction",
            stats.last_q_target_clipped_fraction
        );
        stats.last_terminal_sample_fraction = response.value(
            "last_terminal_sample_fraction",
            stats.last_terminal_sample_fraction
        );
        stats.last_terminal_predicted_q_mean = response.value(
            "last_terminal_predicted_q_mean",
            stats.last_terminal_predicted_q_mean
        );
        stats.last_terminal_td_target_mean = response.value(
            "last_terminal_td_target_mean",
            stats.last_terminal_td_target_mean
        );
        stats.last_critic_disagreement = response.value(
            "last_critic_disagreement",
            stats.last_critic_disagreement
        );
        stats.last_hard_negative_sample_fraction = response.value(
            "last_hard_negative_sample_fraction",
            stats.last_hard_negative_sample_fraction
        );
        stats.parameter_checksum =
            response.value(
                "parameter_checksum", stats.parameter_checksum
            );
        stats.resnet_parameter_checksum = response.value(
            "resnet_parameter_checksum",
            stats.resnet_parameter_checksum
        );
        stats.resnet_training_updates = response.value(
            "resnet_training_updates",
            stats.resnet_training_updates
        );
        stats.checkpoint_loaded =
            response.value(
                "checkpoint_loaded", stats.checkpoint_loaded
            );
        stats.checkpoint_migrated = response.value(
            "checkpoint_migrated", stats.checkpoint_migrated
        );
        stats.resnet_checkpoint_loaded = response.value(
            "resnet_checkpoint_loaded",
            stats.resnet_checkpoint_loaded
        );
        stats.trainable_resnet = response.value(
            "trainable_resnet", stats.trainable_resnet
        );
        stats.device = response.value("device", stats.device);
    }

    void shutdown() noexcept {
        if (service_input != nullptr && service_output != nullptr) {
            try {
                static_cast<void>(transact({{"op", "shutdown"}}));
            } catch (...) {
            }
        }
        if (service_input != nullptr) {
            std::fclose(service_input);
            service_input = nullptr;
        }
        if (service_output != nullptr) {
            std::fclose(service_output);
            service_output = nullptr;
        }
        if (child_pid > 0) {
            int status = 0;
            while (waitpid(child_pid, &status, 0) < 0 && errno == EINTR) {
            }
            child_pid = -1;
        }
    }

    RecurrentPolicyBackendConfig config;
    RecurrentPolicyBackendStats stats;
    pid_t child_pid = -1;
    FILE* service_input = nullptr;
    FILE* service_output = nullptr;
};

RecurrentPolicyBackend::RecurrentPolicyBackend(
    RecurrentPolicyBackendConfig config
) : implementation_(
        std::make_unique<Implementation>(std::move(config))
    ) {}

RecurrentPolicyBackend::~RecurrentPolicyBackend() = default;

void RecurrentPolicyBackend::observe(
    std::span<const float> visual,
    std::span<const std::uint8_t> jpeg,
    std::uint64_t frame_id,
    std::span<const float> controller,
    std::span<const float> context,
    bool episode_started
) {
    auto& state = *implementation_;
    if (
        (
            !state.config.trainable_resnet &&
            visual.size() != state.config.visual_size
        ) ||
        (
            state.config.trainable_resnet &&
            (jpeg.empty() || frame_id == 0)
        ) ||
        controller.size() != state.config.controller_size ||
        context.size() != state.config.context_size
    ) {
        throw std::invalid_argument(
            "trainable recurrent observation shape mismatch"
        );
    }
    nlohmann::json request{
        {"op", "observe"},
        {"controller",
         std::vector<float>(controller.begin(), controller.end())},
        {"context", std::vector<float>(context.begin(), context.end())},
        {"episode_started", episode_started},
        {"frame_id", frame_id},
    };
    if (state.config.trainable_resnet) {
        request["jpeg_base64"] = encode_base64(jpeg);
    } else {
        request["visual"] =
            std::vector<float>(visual.begin(), visual.end());
    }
    static_cast<void>(state.transact(request));
}

std::vector<float> RecurrentPolicyBackend::q_values(
    std::size_t character,
    std::size_t opponent,
    std::size_t action_count
) {
    auto& state = *implementation_;
    const nlohmann::json response = state.transact({
        {"op", "q_values"},
        {"character", character},
        {"opponent", opponent},
        {"action_count", action_count},
    });
    std::vector<float> values =
        response.at("q_values").get<std::vector<float>>();
    if (values.size() != action_count) {
        throw std::runtime_error(
            "recurrent policy returned the wrong Q-value count"
        );
    }
    return values;
}

void RecurrentPolicyBackend::begin_transition(
    std::size_t action,
    std::size_t character,
    std::size_t opponent,
    float selected_q
) {
    static_cast<void>(implementation_->transact({
        {"op", "begin"},
        {"action", action},
        {"character", character},
        {"opponent", opponent},
        {"selected_q", selected_q},
    }));
}

RecurrentPolicyCloseResult RecurrentPolicyBackend::close_transition(
    std::size_t duration,
    float reward,
    bool terminal,
    std::span<const std::uint8_t> next_action_mask
) {
    auto& state = *implementation_;
    const nlohmann::json response = state.transact({
        {"op", "close"},
        {"duration", duration},
        {"reward", reward},
        {"terminal", terminal},
        {"next_action_mask", std::vector<std::uint8_t>(
             next_action_mask.begin(), next_action_mask.end()
         )},
    });
    return {
        .stats = state.stats,
        .training_performed =
            response.value("training_performed", false),
    };
}

void RecurrentPolicyBackend::reset() {
    static_cast<void>(implementation_->transact({{"op", "reset"}}));
}

void RecurrentPolicyBackend::save(
    const std::filesystem::path& path,
    bool include_replay
) {
    auto& state = *implementation_;
    static_cast<void>(state.transact({
        {"op", "save"},
        {"path",
         path.empty()
             ? state.config.checkpoint_path.string()
             : path.string()},
        {"include_replay", include_replay},
    }));
}

const RecurrentPolicyBackendStats& RecurrentPolicyBackend::stats() const {
    return implementation_->stats;
}

const RecurrentPolicyBackendConfig&
RecurrentPolicyBackend::config() const {
    return implementation_->config;
}

}  // namespace gameagent
