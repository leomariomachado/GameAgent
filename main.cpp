#include <baseai/baseai.h>

#include "WebSocketServer.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::atomic_bool stop_requested{false};

void handle_signal(int) {
    stop_requested.store(true);
}

std::string environment_or(const char* name, std::string fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? value : std::move(fallback);
}

std::uint16_t parse_port(std::string_view text) {
    const unsigned long value = std::stoul(std::string(text));
    if (value == 0 || value > 65535) {
        throw std::invalid_argument("port must be between 1 and 65535");
    }
    return static_cast<std::uint16_t>(value);
}

std::size_t parse_megabytes(std::string_view text) {
    const unsigned long value = std::stoul(std::string(text));
    if (value == 0 || value > 1024) {
        throw std::invalid_argument("maximum JPEG size must be between 1 and 1024 MiB");
    }
    return static_cast<std::size_t>(value) * 1024U * 1024U;
}

std::size_t parse_positive_size(std::string_view text, std::string_view name) {
    const unsigned long long value = std::stoull(std::string(text));
    if (value == 0 || value > 1'000'000ULL) {
        throw std::invalid_argument(std::string(name) + " must be between 1 and 1000000");
    }
    return static_cast<std::size_t>(value);
}

int parse_non_negative_int(std::string_view text, std::string_view name) {
    const long value = std::stol(std::string(text));
    if (value < 0 || value > 1024) {
        throw std::invalid_argument(
            std::string(name) + " must be between 0 and 1024"
        );
    }
    return static_cast<int>(value);
}

bool parse_boolean(std::string value, std::string_view name) {
    for (char& character : value) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    throw std::invalid_argument(std::string(name) + " must be 0/1, true/false, yes/no or on/off");
}

bool parse_facing_right(std::string value) {
    for (char& character : value) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    if (value == "right" || value == "direita") {
        return true;
    }
    if (value == "left" || value == "esquerda") {
        return false;
    }
    throw std::invalid_argument(
        "default facing must be right/direita or left/esquerda"
    );
}

bool parse_player_is_left(std::string value) {
    for (char& character : value) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    if (value == "left" || value == "esquerda" || value == "p1") {
        return true;
    }
    if (value == "right" || value == "direita" || value == "p2") {
        return false;
    }
    throw std::invalid_argument(
        "player side must be left/esquerda/P1 or right/direita/P2"
    );
}

float parse_float_range(
    std::string_view text,
    std::string_view name,
    float minimum,
    float maximum
) {
    const float value = std::stof(std::string(text));
    if (!std::isfinite(value) || value < minimum || value > maximum) {
        throw std::invalid_argument(
            std::string(name) + " must be between " +
            std::to_string(minimum) + " and " + std::to_string(maximum)
        );
    }
    return value;
}

void print_help(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n\n"
        << "Options:\n"
        << "  --host HOST             Bind address (default: 0.0.0.0)\n"
        << "  --port PORT             WebSocket port (default: 8080)\n"
        << "  --buffer-frames N       Circular buffer capacity (default: 32)\n"
        << "  --observation-frames N  Frames used per decision (default: 8)\n"
        << "  --action-history N      Confirmed actions given to the agent (default: 8)\n"
        << "  --persistence-queue N   Async JPEG write queue (default: 256)\n"
        << "  --interval-ms MS        Decision interval (default: 50)\n"
        << "  --failsafe-ms MS        Client release timeout advertised by server (default: 200)\n"
        << "  --max-jpeg-mb MB        Maximum JPEG size (default: 8)\n"
        << "  --metrics-path PATH     JSONL metrics path\n"
        << "  --capture-dir PATH      Optional frame capture directory\n"
        << "  --save-frames           Save periodic JPEG samples for debugging\n"
        << "  --debug-capture-ms MS   Debug JPEG interval (default: 3000)\n"
        << "  --collect-training-data Save every accepted frame for offline training\n"
        << "  --training-capture-dir P Training JPEG/metadata directory\n"
        << "  --temporal-model        Enable frozen ResNet50 + GPU LSTM features\n"
        << "  --reinforcement-learning Enable recurrent semi-Markov Double DQN control\n"
        << "  --rl-evaluation         Run RL without updating weights\n"
        << "  --rl-checkpoint-dir P   Shared RL policy checkpoint directory\n"
        << "  --rl-reward-log-ms MS   Accumulated RL reward log interval (default: 5000)\n"
        << "  --rl-no-resume          Start RL weights/replay from scratch\n"
        << "  --no-training-reset     Disable training_reset_state after a lost round\n"
        << "  --player-side SIDE      Controlled HUD side: left/P1 or right/P2\n"
        << "  --resnet50-checkpoint P ImageNet ResNet50 checkpoint path\n"
        << "  --temporal-checkpoint P Trained LSTM/action-head checkpoint\n"
        << "  --device ID             CUDA device used by ResNet50/LSTM (default: 0)\n"
        << "  --lstm-projection N     Visual projection width (default: 256)\n"
        << "  --lstm-hidden N         LSTM hidden width (default: 256)\n"
        << "  --default-facing SIDE   Legacy/default facing: right or left (default: right)\n"
        << "  --seed NUMBER           Deterministic random command seed\n"
        << "  --help                  Show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        gameagent::WebSocketServerOptions server_options;
        server_options.host = environment_or("GAMEAGENT_HOST", "0.0.0.0");
        server_options.port = parse_port(environment_or("GAMEAGENT_PORT", "8080"));
        server_options.frame_buffer_capacity = parse_positive_size(
            environment_or("GAMEAGENT_FRAME_BUFFER_CAPACITY", "32"), "buffer capacity"
        );
        server_options.observation_frames = parse_positive_size(
            environment_or("GAMEAGENT_OBSERVATION_FRAMES", "8"), "observation frames"
        );
        server_options.applied_action_history_size = parse_positive_size(
            environment_or("GAMEAGENT_ACTION_HISTORY_SIZE", "8"), "action history size"
        );
        server_options.persistence_queue_capacity = parse_positive_size(
            environment_or("GAMEAGENT_PERSISTENCE_QUEUE_CAPACITY", "256"),
            "persistence queue capacity"
        );
        server_options.decision_interval = std::chrono::milliseconds(parse_positive_size(
            environment_or("GAMEAGENT_DECISION_INTERVAL_MS", "50"), "decision interval"
        ));
        server_options.failsafe_timeout = std::chrono::milliseconds(parse_positive_size(
            environment_or("GAMEAGENT_FAILSAFE_TIMEOUT_MS", "200"), "failsafe timeout"
        ));
        server_options.max_jpeg_bytes = parse_megabytes(
            environment_or("GAMEAGENT_MAX_JPEG_MB", "8")
        );
        server_options.metrics_path = environment_or(
            "GAMEAGENT_METRICS_PATH", "metrics.jsonl"
        );
        server_options.capture_directory = environment_or(
            "GAMEAGENT_CAPTURE_DIR", "captures"
        );
        server_options.save_frames = parse_boolean(
            environment_or("GAMEAGENT_SAVE_FRAMES", "0"), "frame saving flag"
        );
        server_options.debug_capture_interval =
            std::chrono::milliseconds(parse_positive_size(
                environment_or(
                    "GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS", "3000"
                ),
                "debug capture interval"
            ));
        server_options.collect_training_data = parse_boolean(
            environment_or("GAMEAGENT_COLLECT_TRAINING_DATA", "0"),
            "training data collection flag"
        );
        server_options.training_capture_directory = environment_or(
            "GAMEAGENT_TRAINING_CAPTURE_DIR",
            "/workspace/gameagent/training_captures"
        );
        server_options.temporal_model_enabled = parse_boolean(
            environment_or("GAMEAGENT_TEMPORAL_MODEL", "0"),
            "temporal model flag"
        );
        server_options.reinforcement_learning_enabled = parse_boolean(
            environment_or("GAMEAGENT_REINFORCEMENT_LEARNING", "0"),
            "reinforcement-learning flag"
        );
        server_options.reinforcement_training_enabled = parse_boolean(
            environment_or("GAMEAGENT_RL_TRAINING", "1"),
            "reinforcement-training flag"
        );
        server_options.reinforcement_resume = parse_boolean(
            environment_or("GAMEAGENT_RL_RESUME", "1"),
            "reinforcement-resume flag"
        );
        server_options.reinforcement_trainable_lstm = parse_boolean(
            environment_or("GAMEAGENT_RL_TRAINABLE_LSTM", "1"),
            "trainable-LSTM flag"
        );
        server_options.reinforcement_trainable_resnet = parse_boolean(
            environment_or("GAMEAGENT_RL_TRAINABLE_RESNET", "0"),
            "trainable-ResNet flag"
        );
        server_options.training_reset_state_enabled = parse_boolean(
            environment_or("GAMEAGENT_TRAINING_RESET_STATE", "1"),
            "training-reset-state flag"
        );
        server_options.reinforcement_reward_log_interval =
            std::chrono::milliseconds(parse_positive_size(
                environment_or("GAMEAGENT_RL_REWARD_LOG_INTERVAL_MS", "5000"),
                "RL reward log interval"
            ));
        server_options.player_is_left = parse_player_is_left(
            environment_or("GAMEAGENT_PLAYER_SIDE", "left")
        );
        server_options.resnet50_checkpoint = environment_or(
            "GAMEAGENT_RESNET50_CHECKPOINT",
            "/workspace/resnet50_best.ckpt"
        );
        server_options.temporal_policy_checkpoint = environment_or(
            "GAMEAGENT_TEMPORAL_CHECKPOINT", ""
        );
        server_options.temporal_device_id = parse_non_negative_int(
            environment_or("GAMEAGENT_DEVICE_ID", "0"), "CUDA device"
        );
        server_options.temporal_visual_projection_size = parse_positive_size(
            environment_or("GAMEAGENT_LSTM_PROJECTION_SIZE", "256"),
            "LSTM projection size"
        );
        server_options.temporal_hidden_size = parse_positive_size(
            environment_or("GAMEAGENT_LSTM_HIDDEN_SIZE", "256"),
            "LSTM hidden size"
        );
        server_options.reinforcement_checkpoint_directory = environment_or(
            "GAMEAGENT_RL_CHECKPOINT_DIR",
            "/workspace/gameagent/rl_checkpoints"
        );
        server_options.recurrent_service_script = environment_or(
            "GAMEAGENT_RL_RECURRENT_SERVICE",
            "/tmp/GameAgent/scripts/recurrent_dqn_service.py"
        );
        server_options.reinforcement_network_hidden_size = parse_positive_size(
            environment_or("GAMEAGENT_RL_HIDDEN_SIZE", "128"),
            "RL hidden size"
        );
        server_options.reinforcement_replay_capacity = parse_positive_size(
            environment_or("GAMEAGENT_RL_REPLAY_CAPACITY", "20000"),
            "RL replay capacity"
        );
        server_options.reinforcement_minimum_replay_size = parse_positive_size(
            environment_or("GAMEAGENT_RL_MIN_REPLAY_SIZE", "256"),
            "RL minimum replay size"
        );
        server_options.reinforcement_batch_size = parse_positive_size(
            environment_or("GAMEAGENT_RL_BATCH_SIZE", "64"),
            "RL batch size"
        );
        server_options.reinforcement_target_update_interval = parse_positive_size(
            environment_or("GAMEAGENT_RL_TARGET_UPDATE", "500"),
            "RL target update interval"
        );
        server_options.reinforcement_epsilon_decay_selections = parse_positive_size(
            environment_or("GAMEAGENT_RL_EPSILON_DECAY", "100000"),
            "RL epsilon decay"
        );
        server_options.reinforcement_gamma = parse_float_range(
            environment_or("GAMEAGENT_RL_GAMMA", "0.99"),
            "RL gamma", 0.01f, 1.0f
        );
        server_options.reinforcement_learning_rate = parse_float_range(
            environment_or("GAMEAGENT_RL_LEARNING_RATE", "0.0001"),
            "RL learning rate", 1.0e-7f, 1.0f
        );
        server_options.reinforcement_resnet_learning_rate =
            parse_float_range(
                environment_or(
                    "GAMEAGENT_RL_RESNET_LEARNING_RATE", "0.000001"
                ),
                "RL ResNet learning rate", 1.0e-9f, 1.0e-2f
            );
        server_options.reinforcement_resnet_batch_size =
            parse_positive_size(
                environment_or(
                    "GAMEAGENT_RL_RESNET_BATCH_SIZE", "4"
                ),
                "RL ResNet batch size"
            );
        server_options.reinforcement_resnet_update_interval =
            parse_positive_size(
                environment_or(
                    "GAMEAGENT_RL_RESNET_UPDATE_INTERVAL", "16"
                ),
                "RL ResNet update interval"
            );
        server_options.reinforcement_resnet_replay_images =
            parse_positive_size(
                environment_or(
                    "GAMEAGENT_RL_RESNET_REPLAY_IMAGES", "4096"
                ),
                "RL ResNet replay image limit"
            );
        server_options.reinforcement_epsilon_start = parse_float_range(
            environment_or("GAMEAGENT_RL_EPSILON_START", "1.0"),
            "RL epsilon start", 0.0f, 1.0f
        );
        server_options.reinforcement_epsilon_final = parse_float_range(
            environment_or("GAMEAGENT_RL_EPSILON_FINAL", "0.05"),
            "RL epsilon final", 0.0f, 1.0f
        );
        server_options.reinforcement_scripted_exploration_probability =
            parse_float_range(
                environment_or("GAMEAGENT_RL_SCRIPTED_EXPLORATION", "0.50"),
                "RL scripted exploration probability", 0.0f, 1.0f
            );
        server_options.default_facing_right = parse_facing_right(
            environment_or("GAMEAGENT_DEFAULT_FACING", "right")
        );
        if (const char* token = std::getenv("GAMEAGENT_API_TOKEN");
            token != nullptr && token[0] != '\0') {
            server_options.api_token = token;
        }

        if (const char* seed = std::getenv("GAMEAGENT_RANDOM_SEED");
            seed != nullptr && seed[0] != '\0') {
            server_options.random_seed = static_cast<std::uint32_t>(std::stoul(seed));
        }

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            const auto require_value = [&](std::string_view option) -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(std::string(option) + " requires a value");
                }
                return argv[++i];
            };

            if (argument == "--host") {
                server_options.host = require_value(argument);
            } else if (argument == "--port") {
                server_options.port = parse_port(require_value(argument));
            } else if (argument == "--buffer-frames") {
                server_options.frame_buffer_capacity = parse_positive_size(
                    require_value(argument), "buffer capacity"
                );
            } else if (argument == "--observation-frames") {
                server_options.observation_frames = parse_positive_size(
                    require_value(argument), "observation frames"
                );
            } else if (argument == "--action-history") {
                server_options.applied_action_history_size = parse_positive_size(
                    require_value(argument), "action history size"
                );
            } else if (argument == "--persistence-queue") {
                server_options.persistence_queue_capacity =
                    parse_positive_size(
                        require_value(argument),
                        "persistence queue capacity"
                    );
            } else if (argument == "--interval-ms") {
                server_options.decision_interval = std::chrono::milliseconds(
                    parse_positive_size(require_value(argument), "decision interval")
                );
            } else if (argument == "--failsafe-ms") {
                server_options.failsafe_timeout = std::chrono::milliseconds(
                    parse_positive_size(require_value(argument), "failsafe timeout")
                );
            } else if (argument == "--max-jpeg-mb") {
                server_options.max_jpeg_bytes = parse_megabytes(require_value(argument));
            } else if (argument == "--metrics-path") {
                server_options.metrics_path = require_value(argument);
            } else if (argument == "--capture-dir") {
                server_options.capture_directory = require_value(argument);
            } else if (argument == "--save-frames") {
                server_options.save_frames = true;
            } else if (argument == "--debug-capture-ms") {
                server_options.debug_capture_interval =
                    std::chrono::milliseconds(parse_positive_size(
                        require_value(argument),
                        "debug capture interval"
                    ));
            } else if (argument == "--collect-training-data") {
                server_options.collect_training_data = true;
            } else if (argument == "--training-capture-dir") {
                server_options.training_capture_directory =
                    require_value(argument);
            } else if (argument == "--temporal-model") {
                server_options.temporal_model_enabled = true;
            } else if (argument == "--reinforcement-learning") {
                server_options.reinforcement_learning_enabled = true;
            } else if (argument == "--rl-evaluation") {
                server_options.reinforcement_training_enabled = false;
            } else if (argument == "--rl-checkpoint-dir") {
                server_options.reinforcement_checkpoint_directory =
                    require_value(argument);
            } else if (argument == "--rl-reward-log-ms") {
                server_options.reinforcement_reward_log_interval =
                    std::chrono::milliseconds(parse_positive_size(
                        require_value(argument), "RL reward log interval"
                    ));
            } else if (argument == "--rl-no-resume") {
                server_options.reinforcement_resume = false;
            } else if (argument == "--no-training-reset") {
                server_options.training_reset_state_enabled = false;
            } else if (argument == "--player-side") {
                server_options.player_is_left = parse_player_is_left(
                    require_value(argument)
                );
            } else if (argument == "--resnet50-checkpoint") {
                server_options.resnet50_checkpoint =
                    require_value(argument);
            } else if (argument == "--temporal-checkpoint") {
                server_options.temporal_policy_checkpoint =
                    require_value(argument);
            } else if (argument == "--device") {
                server_options.temporal_device_id =
                    parse_non_negative_int(
                        require_value(argument), "CUDA device"
                    );
            } else if (argument == "--lstm-projection") {
                server_options.temporal_visual_projection_size =
                    parse_positive_size(
                        require_value(argument), "LSTM projection size"
                    );
            } else if (argument == "--lstm-hidden") {
                server_options.temporal_hidden_size =
                    parse_positive_size(
                        require_value(argument), "LSTM hidden size"
                    );
            } else if (argument == "--default-facing") {
                server_options.default_facing_right = parse_facing_right(
                    require_value(argument)
                );
            } else if (argument == "--seed") {
                server_options.random_seed = static_cast<std::uint32_t>(
                    std::stoul(require_value(argument))
                );
            } else if (argument == "--help") {
                print_help(argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("unknown option: " + argument);
            }
        }

        const base::Shape observation_shape{server_options.observation_frames};
        if (base::numel(observation_shape) != server_options.observation_frames) {
            throw std::runtime_error("BaseAI shape validation failed");
        }

        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
#ifdef SIGPIPE
        std::signal(SIGPIPE, SIG_IGN);
#endif

        gameagent::WebSocketServer server(server_options);

        std::cout
            << "GameAgent WebSocket mock server usando BaseAI\n"
            << "observacao: " << base::shape_to_string(observation_shape) << " frames recentes\n"
            << "WebSocket: ws://" << server_options.host << ':'
            << server_options.port << "/ws\n"
            << "health check: http://" << server_options.host << ':'
            << server_options.port << "/health\n"
            << "decisoes: " << (1000.0 / server_options.decision_interval.count()) << " Hz\n"
            << "protocolo de acao: v2/state, failsafe="
            << server_options.failsafe_timeout.count() << " ms\n"
            << "buffer circular: " << server_options.frame_buffer_capacity << " frames\n"
            << "historico confirmado: " << server_options.applied_action_history_size
            << " acoes\n"
            << "perfis: subzero, scorpion, noob_saibot, sindel; "
               "padrao subzero olhando para "
            << (server_options.default_facing_right ? "direita" : "esquerda")
            << '\n'
            << "modelo temporal: "
            << (
                server_options.temporal_model_enabled
                    ? "ResNet50 congelada + LSTM GPU"
                    : "desabilitado"
            )
            << '\n'
            << "agente de reforco: "
            << (
                server_options.reinforcement_learning_enabled
                    ? (server_options.reinforcement_training_enabled
                        ? "Double DQN online (treinamento)"
                        : "Double DQN (avaliacao)")
                    : "desabilitado"
            )
            << '\n'
            << "lado controlado no HUD: "
            << (server_options.player_is_left ? "esquerda/P1" : "direita/P2")
            << '\n'
            << "checkpoint ResNet50: "
            << server_options.resnet50_checkpoint
            << '\n'
            << "checkpoint temporal: "
            << (
                server_options.temporal_policy_checkpoint.empty()
                    ? "nao carregado (teacher-only)"
                    : server_options.temporal_policy_checkpoint.string()
            )
            << '\n'
            << "salvamento de frames: "
            << (
                server_options.save_frames
                    ? "habilitado (debug a cada " +
                          std::to_string(
                              server_options.debug_capture_interval.count()
                          ) +
                          " ms)"
                    : "desabilitado"
            )
            << '\n'
            << "coleta de treinamento: "
            << (
                server_options.collect_training_data
                    ? server_options.training_capture_directory.string()
                    : "desabilitada"
            )
            << '\n'
            << "autenticacao: "
            << (server_options.api_token.has_value() ? "token habilitado" : "desabilitada")
            << '\n'
            << "metricas: " << std::filesystem::absolute(server_options.metrics_path) << '\n';

        return server.run(stop_requested);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
}
