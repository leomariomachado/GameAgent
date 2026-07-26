#include "ReinforcementAgent.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace gameagent {
namespace {

constexpr std::array<char, 8> kCheckpointMagic{
    'G', 'A', 'R', 'L', '0', '0', '0', '1'
};
constexpr std::uint32_t kCheckpointVersion = 1;
constexpr std::uint64_t kCheckpointTrailer = 0x4741524c454e4421ULL;

constexpr int kYHighPunch = 0;
constexpr int kXHighKick = 1;
constexpr int kBLowPunch = 2;
constexpr int kALowKick = 3;
constexpr int kBlock = 4;
constexpr int kRun = 5;
constexpr int kUp = 6;
constexpr int kDown = 7;
constexpr int kLeft = 8;
constexpr int kRight = 9;

struct OptionDefinition {
    std::string name;
    std::vector<std::vector<int>> states;
    bool scripted = false;
};

std::vector<OptionDefinition> make_options(FighterFacing facing) {
    const int forward = facing == FighterFacing::right ? kRight : kLeft;
    const int back = facing == FighterFacing::right ? kLeft : kRight;
    std::vector<OptionDefinition> options{
        {"Neutro", {{}, {}}, false},
        {"Avançar", {{forward}, {forward}, {}}, false},
        {"Recuar", {{back}, {back}, {}}, false},
        {"Pular", {{kUp}, {kUp}, {}}, false},
        {"Agachar", {{kDown}, {kDown}, {}}, false},
        {"Soco alto", {{kYHighPunch}, {kYHighPunch}, {}, {}}, false},
        {"Chute alto", {{kXHighKick}, {kXHighKick}, {}, {}}, false},
        {"Soco baixo", {{kBLowPunch}, {kBLowPunch}, {}, {}}, false},
        {"Chute baixo", {{kALowKick}, {kALowKick}, {}, {}}, false},
        {"Bloquear", {{kBlock}, {kBlock}, {kBlock}, {}}, false},
        {"Corrida curta", {{kRun, forward}, {kRun, forward}, {forward}, {}}, false},
    };
    for (const MockMoveDefinition& move : make_sub_zero_curriculum(facing)) {
        if (!move.states.empty()) {
            options.push_back({move.name, move.states, true});
        }
    }
    return options;
}

std::uint64_t option_hash(const std::vector<OptionDefinition>& options) {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&](std::uint8_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    for (const auto& option : options) {
        for (unsigned char character : option.name) {
            mix(character);
        }
        mix(option.scripted ? 1U : 0U);
        for (const auto& state : option.states) {
            mix(0xffU);
            for (int command : state) {
                mix(static_cast<std::uint8_t>(command));
            }
        }
    }
    return hash;
}

void require_finite(std::span<const float> values, const char* name) {
    if (!std::all_of(values.begin(), values.end(), [](float value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument(std::string(name) + " contains a non-finite value");
    }
}

struct Network {
    std::size_t input_size = 0;
    std::size_t hidden_size = 0;
    std::size_t action_size = 0;
    std::vector<float> w1;
    std::vector<float> b1;
    std::vector<float> w2;
    std::vector<float> b2;

    std::vector<float> forward(
        std::span<const float> input,
        std::vector<float>* hidden_output = nullptr,
        std::vector<float>* preactivation_output = nullptr
    ) const {
        if (input.size() != input_size) {
            throw std::invalid_argument("DQN input has an unexpected size");
        }
        std::vector<float> hidden(hidden_size, 0.0f);
        std::vector<float> preactivation(hidden_size, 0.0f);
        for (std::size_t row = 0; row < hidden_size; ++row) {
            double value = b1[row];
            const std::size_t offset = row * input_size;
            for (std::size_t column = 0; column < input_size; ++column) {
                value += static_cast<double>(w1[offset + column]) * input[column];
            }
            preactivation[row] = static_cast<float>(value);
            hidden[row] = std::max(0.0f, preactivation[row]);
        }
        std::vector<float> output(action_size, 0.0f);
        for (std::size_t row = 0; row < action_size; ++row) {
            double value = b2[row];
            const std::size_t offset = row * hidden_size;
            for (std::size_t column = 0; column < hidden_size; ++column) {
                value += static_cast<double>(w2[offset + column]) * hidden[column];
            }
            output[row] = static_cast<float>(value);
        }
        if (hidden_output != nullptr) {
            *hidden_output = std::move(hidden);
        }
        if (preactivation_output != nullptr) {
            *preactivation_output = std::move(preactivation);
        }
        return output;
    }
};

Network initialize_network(
    std::size_t input_size,
    std::size_t hidden_size,
    std::size_t action_size,
    std::mt19937& random_engine
) {
    Network network{
        .input_size = input_size,
        .hidden_size = hidden_size,
        .action_size = action_size,
        .w1 = std::vector<float>(input_size * hidden_size),
        .b1 = std::vector<float>(hidden_size, 0.0f),
        .w2 = std::vector<float>(hidden_size * action_size),
        .b2 = std::vector<float>(action_size, 0.0f),
    };
    std::normal_distribution<float> first(
        0.0f, std::sqrt(2.0f / static_cast<float>(input_size))
    );
    std::normal_distribution<float> second(
        0.0f, std::sqrt(2.0f / static_cast<float>(hidden_size))
    );
    std::generate(network.w1.begin(), network.w1.end(), [&] {
        return first(random_engine);
    });
    std::generate(network.w2.begin(), network.w2.end(), [&] {
        return second(random_engine);
    });
    return network;
}

struct NetworkVectors {
    std::vector<float> w1;
    std::vector<float> b1;
    std::vector<float> w2;
    std::vector<float> b2;
};

NetworkVectors zero_vectors(const Network& network) {
    return {
        .w1 = std::vector<float>(network.w1.size(), 0.0f),
        .b1 = std::vector<float>(network.b1.size(), 0.0f),
        .w2 = std::vector<float>(network.w2.size(), 0.0f),
        .b2 = std::vector<float>(network.b2.size(), 0.0f),
    };
}

struct ReplayTransition {
    std::vector<float> state;
    std::vector<float> next_state;
    std::uint32_t action = 0;
    std::uint32_t duration = 1;
    float reward = 0.0f;
    bool terminal = false;
};

template <typename Value>
void write_value(std::ostream& output, const Value& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error("failed to write reinforcement checkpoint");
    }
}

template <typename Value>
Value read_value(std::istream& input) {
    Value value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("truncated reinforcement checkpoint");
    }
    return value;
}

void write_vector(std::ostream& output, const std::vector<float>& values) {
    write_value(output, static_cast<std::uint64_t>(values.size()));
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!output) {
        throw std::runtime_error("failed to write reinforcement checkpoint vector");
    }
}

std::vector<float> read_vector(std::istream& input, std::size_t expected_size) {
    const std::uint64_t size = read_value<std::uint64_t>(input);
    if (size != expected_size) {
        throw std::runtime_error("reinforcement checkpoint tensor size mismatch");
    }
    std::vector<float> values(expected_size);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!input) {
        throw std::runtime_error("truncated reinforcement checkpoint tensor");
    }
    require_finite(values, "checkpoint tensor");
    return values;
}

void write_network(std::ostream& output, const Network& network) {
    write_vector(output, network.w1);
    write_vector(output, network.b1);
    write_vector(output, network.w2);
    write_vector(output, network.b2);
}

void read_network(std::istream& input, Network& network) {
    network.w1 = read_vector(input, network.w1.size());
    network.b1 = read_vector(input, network.b1.size());
    network.w2 = read_vector(input, network.w2.size());
    network.b2 = read_vector(input, network.b2.size());
}

void write_vectors(std::ostream& output, const NetworkVectors& vectors) {
    write_vector(output, vectors.w1);
    write_vector(output, vectors.b1);
    write_vector(output, vectors.w2);
    write_vector(output, vectors.b2);
}

void read_vectors(
    std::istream& input,
    NetworkVectors& vectors
) {
    vectors.w1 = read_vector(input, vectors.w1.size());
    vectors.b1 = read_vector(input, vectors.b1.size());
    vectors.w2 = read_vector(input, vectors.w2.size());
    vectors.b2 = read_vector(input, vectors.b2.size());
}

}  // namespace

struct ReinforcementAgent::Implementation {
    explicit Implementation(ReinforcementAgentConfig value)
        : config(std::move(value)),
          options(make_options(config.facing)),
          random_engine(config.random_seed),
          online(initialize_network(
              config.input_size,
              config.network_hidden_size,
              options.size(),
              random_engine
          )),
          target(online),
          first_moment(zero_vectors(online)),
          second_moment(zero_vectors(online)) {
        validate_config();
        for (std::size_t index = 0; index < options.size(); ++index) {
            (options[index].scripted ? scripted_options : primitive_options)
                .push_back(index);
        }
        if (config.resume && !config.checkpoint_path.empty() &&
            std::filesystem::is_regular_file(config.checkpoint_path)) {
            load(config.checkpoint_path);
            checkpoint_loaded = true;
        }
    }

    void validate_config() const {
        if (config.input_size == 0 || config.network_hidden_size == 0 ||
            config.replay_capacity == 0 || config.batch_size == 0 ||
            config.minimum_replay_size == 0 ||
            config.training_updates_per_transition == 0 ||
            config.target_update_interval == 0 ||
            config.gamma <= 0.0f || config.gamma > 1.0f ||
            config.learning_rate <= 0.0f ||
            config.adam_beta1 <= 0.0f || config.adam_beta1 >= 1.0f ||
            config.adam_beta2 <= 0.0f || config.adam_beta2 >= 1.0f ||
            config.adam_epsilon <= 0.0f || config.gradient_clip <= 0.0f ||
            config.epsilon_start < 0.0f || config.epsilon_start > 1.0f ||
            config.epsilon_final < 0.0f || config.epsilon_final > 1.0f ||
            config.epsilon_final > config.epsilon_start ||
            config.epsilon_decay_selections == 0 ||
            config.scripted_exploration_probability < 0.0f ||
            config.scripted_exploration_probability > 1.0f) {
            throw std::invalid_argument("invalid reinforcement-agent configuration");
        }
    }

    float epsilon() const {
        const double progress = std::min(
            1.0,
            static_cast<double>(action_selections) /
                static_cast<double>(config.epsilon_decay_selections)
        );
        return static_cast<float>(
            config.epsilon_start +
            (config.epsilon_final - config.epsilon_start) * progress
        );
    }

    std::pair<std::size_t, std::string> select_action(
        std::span<const float> state,
        float& selected_q
    ) {
        const std::vector<float> values = online.forward(state);
        std::uniform_real_distribution<float> probability(0.0f, 1.0f);
        std::size_t selected = 0;
        std::string source = "dqn_exploit";
        if (probability(random_engine) < epsilon()) {
            const bool choose_scripted =
                probability(random_engine) <
                    config.scripted_exploration_probability;
            const auto& candidates = choose_scripted
                ? scripted_options
                : primitive_options;
            std::uniform_int_distribution<std::size_t> distribution(
                0, candidates.size() - 1U
            );
            selected = candidates[distribution(random_engine)];
            source = choose_scripted
                ? "epsilon_scripted"
                : "epsilon_primitive";
        } else {
            selected = static_cast<std::size_t>(std::distance(
                values.begin(),
                std::max_element(values.begin(), values.end())
            ));
        }
        selected_q = values[selected];
        ++action_selections;
        return {selected, std::move(source)};
    }

    void append_replay(ReplayTransition transition) {
        if (replay.size() < config.replay_capacity) {
            replay.push_back(std::move(transition));
        } else {
            replay[replay_next] = std::move(transition);
            replay_next = (replay_next + 1U) % config.replay_capacity;
        }
    }

    static void accumulate_gradient(
        const Network& network,
        std::span<const float> state,
        std::size_t action,
        float target_value,
        NetworkVectors& gradient,
        double& loss
    ) {
        std::vector<float> hidden;
        std::vector<float> preactivation;
        const std::vector<float> q = network.forward(
            state, &hidden, &preactivation
        );
        const float error = q[action] - target_value;
        const float absolute_error = std::abs(error);
        loss += absolute_error <= 1.0f
            ? 0.5 * static_cast<double>(error) * error
            : static_cast<double>(absolute_error) - 0.5;
        const float output_gradient = std::clamp(error, -1.0f, 1.0f);
        gradient.b2[action] += output_gradient;
        const std::size_t w2_offset = action * network.hidden_size;
        for (std::size_t hidden_index = 0;
             hidden_index < network.hidden_size;
             ++hidden_index) {
            gradient.w2[w2_offset + hidden_index] +=
                output_gradient * hidden[hidden_index];
            if (preactivation[hidden_index] <= 0.0f) {
                continue;
            }
            const float hidden_gradient =
                output_gradient * network.w2[w2_offset + hidden_index];
            gradient.b1[hidden_index] += hidden_gradient;
            const std::size_t w1_offset =
                hidden_index * network.input_size;
            for (std::size_t input_index = 0;
                 input_index < network.input_size;
                 ++input_index) {
                gradient.w1[w1_offset + input_index] +=
                    hidden_gradient * state[input_index];
            }
        }
    }

    void adam_update_vector(
        std::vector<float>& parameter,
        std::vector<float>& first,
        std::vector<float>& second,
        const std::vector<float>& gradient,
        float inverse_batch
    ) {
        const double beta1_power = std::pow(
            static_cast<double>(config.adam_beta1),
            static_cast<double>(training_updates + 1U)
        );
        const double beta2_power = std::pow(
            static_cast<double>(config.adam_beta2),
            static_cast<double>(training_updates + 1U)
        );
        for (std::size_t index = 0; index < parameter.size(); ++index) {
            const float value = std::clamp(
                gradient[index] * inverse_batch,
                -config.gradient_clip,
                config.gradient_clip
            );
            first[index] =
                config.adam_beta1 * first[index] +
                (1.0f - config.adam_beta1) * value;
            second[index] =
                config.adam_beta2 * second[index] +
                (1.0f - config.adam_beta2) * value * value;
            const double corrected_first =
                first[index] / (1.0 - beta1_power);
            const double corrected_second =
                second[index] / (1.0 - beta2_power);
            parameter[index] -= config.learning_rate *
                static_cast<float>(
                    corrected_first /
                    (std::sqrt(corrected_second) + config.adam_epsilon)
                );
        }
    }

    bool train_once() {
        const std::size_t minimum = std::max(
            config.minimum_replay_size, config.batch_size
        );
        if (!config.training_enabled || replay.size() < minimum) {
            return false;
        }
        NetworkVectors gradient = zero_vectors(online);
        std::uniform_int_distribution<std::size_t> distribution(
            0, replay.size() - 1U
        );
        double loss = 0.0;
        for (std::size_t sample = 0; sample < config.batch_size; ++sample) {
            const ReplayTransition& transition = replay[distribution(random_engine)];
            float target_value = transition.reward;
            if (!transition.terminal) {
                const std::vector<float> online_next =
                    online.forward(transition.next_state);
                const std::size_t best_action =
                    static_cast<std::size_t>(std::distance(
                        online_next.begin(),
                        std::max_element(
                            online_next.begin(), online_next.end()
                        )
                    ));
                const std::vector<float> target_next =
                    target.forward(transition.next_state);
                target_value +=
                    std::pow(config.gamma, static_cast<float>(transition.duration)) *
                    target_next[best_action];
            }
            accumulate_gradient(
                online,
                transition.state,
                transition.action,
                target_value,
                gradient,
                loss
            );
        }
        const float inverse_batch =
            1.0f / static_cast<float>(config.batch_size);
        adam_update_vector(
            online.w1, first_moment.w1, second_moment.w1,
            gradient.w1, inverse_batch
        );
        adam_update_vector(
            online.b1, first_moment.b1, second_moment.b1,
            gradient.b1, inverse_batch
        );
        adam_update_vector(
            online.w2, first_moment.w2, second_moment.w2,
            gradient.w2, inverse_batch
        );
        adam_update_vector(
            online.b2, first_moment.b2, second_moment.b2,
            gradient.b2, inverse_batch
        );
        ++training_updates;
        last_training_loss = static_cast<float>(
            loss / static_cast<double>(config.batch_size)
        );
        if (training_updates % config.target_update_interval == 0U) {
            target = online;
        }
        return true;
    }

    bool close_active_transition(
        std::span<const float> next_state,
        bool terminal
    ) {
        if (!active_option.has_value()) {
            return false;
        }
        if (option_start_state.size() != config.input_size) {
            active_option.reset();
            return false;
        }
        std::vector<float> next;
        if (next_state.size() == config.input_size) {
            next.assign(next_state.begin(), next_state.end());
        } else {
            next = option_start_state;
        }
        append_replay({
            .state = option_start_state,
            .next_state = std::move(next),
            .action = static_cast<std::uint32_t>(*active_option),
            .duration = static_cast<std::uint32_t>(
                std::max<std::size_t>(1U, option_elapsed_steps)
            ),
            .reward = option_reward,
            .terminal = terminal,
        });
        bool trained = false;
        for (std::size_t update = 0;
             update < config.training_updates_per_transition;
             ++update) {
            trained = train_once() || trained;
        }
        active_option.reset();
        active_step = 0;
        option_start_state.clear();
        option_reward = 0.0f;
        option_reward_discount = 1.0f;
        option_elapsed_steps = 0;
        return trained;
    }

    void write_checkpoint(const std::filesystem::path& path) const {
        if (path.empty()) {
            return;
        }
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path());
        }
        const std::filesystem::path temporary = path.string() + ".tmp";
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "could not create reinforcement checkpoint: " +
                temporary.string()
            );
        }
        output.write(kCheckpointMagic.data(), kCheckpointMagic.size());
        write_value(output, kCheckpointVersion);
        write_value(output, static_cast<std::uint64_t>(config.input_size));
        write_value(output, static_cast<std::uint64_t>(config.network_hidden_size));
        write_value(output, static_cast<std::uint64_t>(options.size()));
        write_value(output, option_hash(options));
        write_value(output, action_selections);
        write_value(output, training_updates);
        write_value(output, episodes_completed);
        write_value(output, current_episode_return);
        write_value(output, best_episode_return);
        write_value(output, last_training_loss);
        write_network(output, online);
        write_network(output, target);
        write_vectors(output, first_moment);
        write_vectors(output, second_moment);
        const std::uint64_t replay_count = config.save_replay
            ? static_cast<std::uint64_t>(replay.size())
            : 0U;
        write_value(output, replay_count);
        write_value(output, static_cast<std::uint64_t>(replay_next));
        for (std::size_t index = 0;
             index < static_cast<std::size_t>(replay_count);
             ++index) {
            const ReplayTransition& transition = replay[index];
            write_vector(output, transition.state);
            write_vector(output, transition.next_state);
            write_value(output, transition.action);
            write_value(output, transition.duration);
            write_value(output, transition.reward);
            write_value(output, static_cast<std::uint8_t>(transition.terminal));
        }
        write_value(output, kCheckpointTrailer);
        output.flush();
        if (!output) {
            throw std::runtime_error("failed to flush reinforcement checkpoint");
        }
        output.close();
        std::filesystem::rename(temporary, path);
    }

    void load(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "could not open reinforcement checkpoint: " + path.string()
            );
        }
        std::array<char, 8> magic{};
        input.read(magic.data(), magic.size());
        if (!input || magic != kCheckpointMagic ||
            read_value<std::uint32_t>(input) != kCheckpointVersion) {
            throw std::runtime_error("invalid reinforcement checkpoint header");
        }
        const auto input_size = read_value<std::uint64_t>(input);
        const auto hidden_size = read_value<std::uint64_t>(input);
        const auto action_size = read_value<std::uint64_t>(input);
        const auto saved_option_hash = read_value<std::uint64_t>(input);
        if (input_size != config.input_size ||
            hidden_size != config.network_hidden_size ||
            action_size != options.size() ||
            saved_option_hash != option_hash(options)) {
            throw std::runtime_error(
                "reinforcement checkpoint architecture/options mismatch"
            );
        }
        action_selections = read_value<std::uint64_t>(input);
        training_updates = read_value<std::uint64_t>(input);
        episodes_completed = read_value<std::uint64_t>(input);
        current_episode_return = read_value<float>(input);
        best_episode_return = read_value<float>(input);
        last_training_loss = read_value<float>(input);
        read_network(input, online);
        read_network(input, target);
        read_vectors(input, first_moment);
        read_vectors(input, second_moment);
        const std::uint64_t replay_count = read_value<std::uint64_t>(input);
        replay_next = static_cast<std::size_t>(
            read_value<std::uint64_t>(input)
        );
        if (replay_count > config.replay_capacity) {
            throw std::runtime_error("reinforcement checkpoint replay is too large");
        }
        replay.clear();
        replay.reserve(static_cast<std::size_t>(replay_count));
        for (std::uint64_t index = 0; index < replay_count; ++index) {
            ReplayTransition transition;
            transition.state = read_vector(input, config.input_size);
            transition.next_state = read_vector(input, config.input_size);
            transition.action = read_value<std::uint32_t>(input);
            transition.duration = read_value<std::uint32_t>(input);
            transition.reward = read_value<float>(input);
            transition.terminal = read_value<std::uint8_t>(input) != 0;
            if (transition.action >= options.size() ||
                transition.duration == 0 ||
                !std::isfinite(transition.reward)) {
                throw std::runtime_error("invalid reinforcement replay transition");
            }
            replay.push_back(std::move(transition));
        }
        if (replay.size() < config.replay_capacity) {
            replay_next = 0;
        } else if (replay_next >= replay.size()) {
            throw std::runtime_error("invalid reinforcement replay cursor");
        }
        if (read_value<std::uint64_t>(input) != kCheckpointTrailer) {
            throw std::runtime_error("reinforcement checkpoint trailer mismatch");
        }
    }

    ReinforcementAgentConfig config;
    std::vector<OptionDefinition> options;
    std::vector<std::size_t> primitive_options;
    std::vector<std::size_t> scripted_options;
    std::mt19937 random_engine;
    Network online;
    Network target;
    NetworkVectors first_moment;
    NetworkVectors second_moment;
    std::vector<ReplayTransition> replay;
    std::size_t replay_next = 0;
    std::optional<std::size_t> active_option;
    std::size_t active_step = 0;
    std::vector<float> option_start_state;
    float option_reward = 0.0f;
    float option_reward_discount = 1.0f;
    std::size_t option_elapsed_steps = 0;
    std::string active_selection_source;
    float active_selected_q = 0.0f;
    std::uint64_t action_selections = 0;
    std::uint64_t training_updates = 0;
    std::uint64_t episodes_completed = 0;
    std::uint64_t current_episode_id = 0;
    float current_episode_return = 0.0f;
    float best_episode_return = -std::numeric_limits<float>::infinity();
    float last_training_loss = 0.0f;
    bool checkpoint_loaded = false;
};

ReinforcementAgent::ReinforcementAgent(ReinforcementAgentConfig config)
    : implementation_(
          std::make_unique<Implementation>(std::move(config))
      ) {}

ReinforcementAgent::~ReinforcementAgent() = default;
ReinforcementAgent::ReinforcementAgent(ReinforcementAgent&&) noexcept = default;
ReinforcementAgent& ReinforcementAgent::operator=(
    ReinforcementAgent&&
) noexcept = default;

ReinforcementDecision ReinforcementAgent::step(
    const ReinforcementStepInput& input
) {
    auto& state = *implementation_;
    if (!input.state.empty()) {
        if (input.state.size() != state.config.input_size) {
            throw std::invalid_argument("reinforcement step state size mismatch");
        }
        require_finite(input.state, "reinforcement state");
    }
    if (!std::isfinite(input.reward)) {
        throw std::invalid_argument("reinforcement reward is not finite");
    }

    ReinforcementDecision decision;
    decision.reward_consumed = input.reward;
    if (input.episode_started ||
        (input.episode_id != 0 && input.episode_id != state.current_episode_id)) {
        state.current_episode_id = input.episode_id;
        state.current_episode_return = 0.0f;
    }
    state.current_episode_return += input.reward;
    if (state.active_option.has_value()) {
        state.option_reward += state.option_reward_discount * input.reward;
        state.option_reward_discount *= state.config.gamma;
    }

    bool trained = false;
    const bool option_complete =
        state.active_option.has_value() &&
        state.active_step >= state.options[*state.active_option].states.size();
    if (input.episode_ended || option_complete) {
        trained = state.close_active_transition(
            input.state,
            input.episode_ended
        );
    }

    if (input.episode_ended) {
        ++state.episodes_completed;
        const bool new_best =
            state.current_episode_return > state.best_episode_return;
        if (new_best) {
            state.best_episode_return = state.current_episode_return;
        }
        if (!state.config.checkpoint_path.empty()) {
            state.write_checkpoint(state.config.checkpoint_path);
            decision.checkpoint_saved = true;
        }
        if (new_best && !state.config.best_checkpoint_path.empty()) {
            state.write_checkpoint(state.config.best_checkpoint_path);
        }
    }

    if (input.action_allowed && input.state.size() == state.config.input_size) {
        if (!state.active_option.has_value()) {
            float q = 0.0f;
            auto [selected, source] = state.select_action(input.state, q);
            state.active_option = selected;
            state.active_step = 0;
            state.option_start_state.assign(input.state.begin(), input.state.end());
            state.option_reward = 0.0f;
            state.option_reward_discount = 1.0f;
            state.option_elapsed_steps = 0;
            state.active_selection_source = std::move(source);
            state.active_selected_q = q;
        }
        const std::size_t selected = *state.active_option;
        const OptionDefinition& option = state.options[selected];
        decision.commands = option.states[state.active_step];
        decision.option_name = option.name;
        decision.selection_source = state.active_selection_source;
        decision.option_index = selected;
        decision.option_step = state.active_step + 1U;
        decision.option_steps = option.states.size();
        decision.selected_q = state.active_selected_q;
        ++state.active_step;
        ++state.option_elapsed_steps;
    }

    decision.epsilon = state.epsilon();
    decision.episode_return = state.current_episode_return;
    decision.training_loss = state.last_training_loss;
    decision.replay_size = state.replay.size();
    decision.training_updates = state.training_updates;
    decision.training_performed = trained;
    return decision;
}

void ReinforcementAgent::reset_episode(bool save_checkpoint) {
    auto& state = *implementation_;
    state.active_option.reset();
    state.active_step = 0;
    state.option_start_state.clear();
    state.option_reward = 0.0f;
    state.option_reward_discount = 1.0f;
    state.option_elapsed_steps = 0;
    state.current_episode_return = 0.0f;
    state.current_episode_id = 0;
    if (save_checkpoint && !state.config.checkpoint_path.empty()) {
        state.write_checkpoint(state.config.checkpoint_path);
    }
}

void ReinforcementAgent::save_checkpoint(
    const std::filesystem::path& path
) {
    const auto& selected = path.empty()
        ? implementation_->config.checkpoint_path
        : path;
    implementation_->write_checkpoint(selected);
}

void ReinforcementAgent::load_checkpoint(
    const std::filesystem::path& path
) {
    implementation_->load(path);
    implementation_->checkpoint_loaded = true;
}

ReinforcementAgentStats ReinforcementAgent::stats() const {
    const auto& state = *implementation_;
    return {
        .action_selections = state.action_selections,
        .training_updates = state.training_updates,
        .episodes_completed = state.episodes_completed,
        .replay_size = state.replay.size(),
        .epsilon = state.epsilon(),
        .last_training_loss = state.last_training_loss,
        .current_episode_return = state.current_episode_return,
        .best_episode_return = state.best_episode_return,
        .checkpoint_loaded = state.checkpoint_loaded,
    };
}

std::vector<std::string> ReinforcementAgent::option_names() const {
    std::vector<std::string> names;
    names.reserve(implementation_->options.size());
    for (const auto& option : implementation_->options) {
        names.push_back(option.name);
    }
    return names;
}

const ReinforcementAgentConfig& ReinforcementAgent::config() const {
    return implementation_->config;
}

}  // namespace gameagent
