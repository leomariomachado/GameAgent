#include "ReinforcementAgent.h"
#include "RecurrentPolicyBackend.h"

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
    'G', 'A', 'R', 'L', '0', '0', '0', '4'
};
constexpr std::uint32_t kCheckpointVersion = 4;
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

constexpr std::array<FighterCharacter, 4> kControlledCharacters{
    FighterCharacter::subzero,
    FighterCharacter::scorpion,
    FighterCharacter::noob_saibot,
    FighterCharacter::sindel,
};
constexpr std::size_t kControlledCharacterCount =
    kControlledCharacters.size();

std::size_t character_index(FighterCharacter character) {
    const auto found = std::find(
        kControlledCharacters.begin(),
        kControlledCharacters.end(),
        character
    );
    if (found == kControlledCharacters.end()) {
        throw std::invalid_argument("unsupported controlled character");
    }
    return static_cast<std::size_t>(
        std::distance(kControlledCharacters.begin(), found)
    );
}

std::size_t opponent_index(OpponentCharacter character) {
    const std::size_t index = static_cast<std::size_t>(character);
    if (index >= kOpponentCharacterFeatureCount) {
        throw std::invalid_argument("unsupported opponent character");
    }
    return index;
}

struct OptionDefinition {
    std::string name;
    std::vector<std::vector<int>> states;
    bool scripted = false;
};

std::vector<OptionDefinition> make_options(FighterCharacter character) {
    // Checkpoint action programs are canonical. The complete program is
    // mirrored only when emitted by a left-facing session.
    constexpr int forward = kRight;
    constexpr int back = kLeft;
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
        {
            "Corrida curta",
            {{kRun, forward}, {kRun, forward}, {forward}, {}},
            false
        },
    };
    for (const MockMoveDefinition& move :
         make_character_curriculum(character, FighterFacing::right)) {
        if (!move.states.empty()) {
            options.push_back({move.name, move.states, true});
        }
    }
    return options;
}

using CharacterOptions =
    std::array<std::vector<OptionDefinition>, kControlledCharacterCount>;

CharacterOptions make_all_options() {
    CharacterOptions result;
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = make_options(kControlledCharacters[index]);
    }
    return result;
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
        throw std::invalid_argument(
            std::string(name) + " contains a non-finite value"
        );
    }
}

struct PolicyHead {
    std::size_t action_size = 0;
    std::vector<float> weights;
    std::vector<float> bias;
};

struct SharedPolicyNetwork {
    std::size_t state_size = 0;
    std::size_t character_embedding_size = 0;
    std::size_t opponent_embedding_size = 0;
    std::size_t hidden_size = 0;
    std::vector<float> character_embeddings;
    std::vector<float> opponent_embeddings;
    std::vector<float> shared_weights;
    std::vector<float> shared_bias;
    std::array<PolicyHead, kControlledCharacterCount> heads;
};

struct SharedPolicyVectors {
    std::vector<float> character_embeddings;
    std::vector<float> opponent_embeddings;
    std::vector<float> shared_weights;
    std::vector<float> shared_bias;
    std::array<PolicyHead, kControlledCharacterCount> heads;
};

SharedPolicyNetwork initialize_network(
    const ReinforcementAgentConfig& config,
    const CharacterOptions& options,
    std::mt19937& random_engine
) {
    const std::size_t augmented_size =
        config.input_size +
        config.character_embedding_size +
        config.opponent_embedding_size;
    SharedPolicyNetwork network{
        .state_size = config.input_size,
        .character_embedding_size = config.character_embedding_size,
        .opponent_embedding_size = config.opponent_embedding_size,
        .hidden_size = config.network_hidden_size,
        .character_embeddings = std::vector<float>(
            kControlledCharacterCount * config.character_embedding_size
        ),
        .opponent_embeddings = std::vector<float>(
            kOpponentCharacterFeatureCount * config.opponent_embedding_size
        ),
        .shared_weights = std::vector<float>(
            augmented_size * config.network_hidden_size
        ),
        .shared_bias = std::vector<float>(
            config.network_hidden_size, 0.0f
        ),
    };

    std::normal_distribution<float> embedding_distribution(
        0.0f,
        1.0f / std::sqrt(
            static_cast<float>(
                std::max<std::size_t>(
                    config.character_embedding_size,
                    config.opponent_embedding_size
                )
            )
        )
    );
    std::normal_distribution<float> shared_distribution(
        0.0f,
        std::sqrt(2.0f / static_cast<float>(augmented_size))
    );
    std::normal_distribution<float> head_distribution(
        0.0f,
        std::sqrt(
            2.0f / static_cast<float>(config.network_hidden_size)
        )
    );
    std::generate(
        network.character_embeddings.begin(),
        network.character_embeddings.end(),
        [&] { return embedding_distribution(random_engine); }
    );
    std::generate(
        network.opponent_embeddings.begin(),
        network.opponent_embeddings.end(),
        [&] { return embedding_distribution(random_engine); }
    );
    std::generate(
        network.shared_weights.begin(),
        network.shared_weights.end(),
        [&] { return shared_distribution(random_engine); }
    );
    for (std::size_t index = 0; index < network.heads.size(); ++index) {
        PolicyHead& head = network.heads[index];
        head.action_size = options[index].size();
        head.weights.resize(
            head.action_size * config.network_hidden_size
        );
        head.bias.assign(head.action_size, 0.0f);
        std::generate(
            head.weights.begin(),
            head.weights.end(),
            [&] { return head_distribution(random_engine); }
        );
    }
    return network;
}

SharedPolicyVectors zero_vectors(const SharedPolicyNetwork& network) {
    SharedPolicyVectors vectors{
        .character_embeddings =
            std::vector<float>(
                network.character_embeddings.size(), 0.0f
            ),
        .opponent_embeddings =
            std::vector<float>(
                network.opponent_embeddings.size(), 0.0f
            ),
        .shared_weights =
            std::vector<float>(network.shared_weights.size(), 0.0f),
        .shared_bias =
            std::vector<float>(network.shared_bias.size(), 0.0f),
    };
    for (std::size_t index = 0; index < vectors.heads.size(); ++index) {
        vectors.heads[index].action_size =
            network.heads[index].action_size;
        vectors.heads[index].weights.assign(
            network.heads[index].weights.size(), 0.0f
        );
        vectors.heads[index].bias.assign(
            network.heads[index].bias.size(), 0.0f
        );
    }
    return vectors;
}

std::vector<float> make_augmented_input(
    const SharedPolicyNetwork& network,
    std::span<const float> state,
    FighterCharacter character,
    OpponentCharacter opponent
) {
    if (state.size() != network.state_size) {
        throw std::invalid_argument("DQN state has an unexpected size");
    }
    const std::size_t player = character_index(character);
    const std::size_t opponent_value = opponent_index(opponent);
    std::vector<float> augmented;
    augmented.reserve(
        state.size() +
        network.character_embedding_size +
        network.opponent_embedding_size
    );
    augmented.insert(augmented.end(), state.begin(), state.end());
    const auto character_begin =
        network.character_embeddings.begin() +
        static_cast<std::ptrdiff_t>(
            player * network.character_embedding_size
        );
    augmented.insert(
        augmented.end(),
        character_begin,
        character_begin +
            static_cast<std::ptrdiff_t>(
                network.character_embedding_size
            )
    );
    const auto opponent_begin =
        network.opponent_embeddings.begin() +
        static_cast<std::ptrdiff_t>(
            opponent_value * network.opponent_embedding_size
        );
    augmented.insert(
        augmented.end(),
        opponent_begin,
        opponent_begin +
            static_cast<std::ptrdiff_t>(
                network.opponent_embedding_size
            )
    );
    return augmented;
}

std::vector<float> forward(
    const SharedPolicyNetwork& network,
    std::span<const float> state,
    FighterCharacter character,
    OpponentCharacter opponent,
    std::vector<float>* augmented_output = nullptr,
    std::vector<float>* hidden_output = nullptr,
    std::vector<float>* preactivation_output = nullptr
) {
    std::vector<float> augmented = make_augmented_input(
        network, state, character, opponent
    );
    std::vector<float> hidden(network.hidden_size, 0.0f);
    std::vector<float> preactivation(network.hidden_size, 0.0f);
    for (std::size_t row = 0; row < network.hidden_size; ++row) {
        double value = network.shared_bias[row];
        const std::size_t offset = row * augmented.size();
        for (std::size_t column = 0; column < augmented.size(); ++column) {
            value +=
                static_cast<double>(
                    network.shared_weights[offset + column]
                ) *
                augmented[column];
        }
        preactivation[row] = static_cast<float>(value);
        hidden[row] = std::max(0.0f, preactivation[row]);
    }

    const PolicyHead& head = network.heads[character_index(character)];
    std::vector<float> output(head.action_size, 0.0f);
    for (std::size_t row = 0; row < head.action_size; ++row) {
        double value = head.bias[row];
        const std::size_t offset = row * network.hidden_size;
        for (
            std::size_t column = 0;
            column < network.hidden_size;
            ++column
        ) {
            value +=
                static_cast<double>(head.weights[offset + column]) *
                hidden[column];
        }
        output[row] = static_cast<float>(value);
    }
    if (augmented_output != nullptr) {
        *augmented_output = std::move(augmented);
    }
    if (hidden_output != nullptr) {
        *hidden_output = std::move(hidden);
    }
    if (preactivation_output != nullptr) {
        *preactivation_output = std::move(preactivation);
    }
    return output;
}

struct ReplayTransition {
    std::vector<float> state;
    std::vector<float> next_state;
    std::uint32_t action = 0;
    std::uint32_t duration = 1;
    float reward = 0.0f;
    bool terminal = false;
    FighterCharacter character = FighterCharacter::subzero;
    OpponentCharacter opponent = OpponentCharacter::unknown;
};

template <typename Value>
void write_value(std::ostream& output, const Value& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error(
            "failed to write reinforcement checkpoint"
        );
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
        throw std::runtime_error(
            "failed to write reinforcement checkpoint vector"
        );
    }
}

std::vector<float> read_vector(
    std::istream& input,
    std::size_t expected_size
) {
    const std::uint64_t size = read_value<std::uint64_t>(input);
    if (size != expected_size) {
        throw std::runtime_error(
            "reinforcement checkpoint tensor size mismatch"
        );
    }
    std::vector<float> values(expected_size);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!input) {
        throw std::runtime_error(
            "truncated reinforcement checkpoint tensor"
        );
    }
    require_finite(values, "checkpoint tensor");
    return values;
}

void write_network(
    std::ostream& output,
    const SharedPolicyNetwork& network
) {
    write_vector(output, network.character_embeddings);
    write_vector(output, network.opponent_embeddings);
    write_vector(output, network.shared_weights);
    write_vector(output, network.shared_bias);
    for (const PolicyHead& head : network.heads) {
        write_vector(output, head.weights);
        write_vector(output, head.bias);
    }
}

void read_network(
    std::istream& input,
    SharedPolicyNetwork& network
) {
    network.character_embeddings =
        read_vector(input, network.character_embeddings.size());
    network.opponent_embeddings =
        read_vector(input, network.opponent_embeddings.size());
    network.shared_weights =
        read_vector(input, network.shared_weights.size());
    network.shared_bias =
        read_vector(input, network.shared_bias.size());
    for (PolicyHead& head : network.heads) {
        head.weights = read_vector(input, head.weights.size());
        head.bias = read_vector(input, head.bias.size());
    }
}

void write_vectors(
    std::ostream& output,
    const SharedPolicyVectors& vectors
) {
    write_vector(output, vectors.character_embeddings);
    write_vector(output, vectors.opponent_embeddings);
    write_vector(output, vectors.shared_weights);
    write_vector(output, vectors.shared_bias);
    for (const PolicyHead& head : vectors.heads) {
        write_vector(output, head.weights);
        write_vector(output, head.bias);
    }
}

void read_vectors(
    std::istream& input,
    SharedPolicyVectors& vectors
) {
    vectors.character_embeddings =
        read_vector(input, vectors.character_embeddings.size());
    vectors.opponent_embeddings =
        read_vector(input, vectors.opponent_embeddings.size());
    vectors.shared_weights =
        read_vector(input, vectors.shared_weights.size());
    vectors.shared_bias =
        read_vector(input, vectors.shared_bias.size());
    for (PolicyHead& head : vectors.heads) {
        head.weights = read_vector(input, head.weights.size());
        head.bias = read_vector(input, head.bias.size());
    }
}

double shared_parameter_checksum(const SharedPolicyNetwork& network) {
    long double result = 0.0;
    const auto accumulate = [&](const std::vector<float>& values) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            result += static_cast<long double>(values[index]) *
                static_cast<long double>((index % 97U) + 1U);
        }
    };
    accumulate(network.character_embeddings);
    accumulate(network.opponent_embeddings);
    accumulate(network.shared_weights);
    accumulate(network.shared_bias);
    return static_cast<double>(result);
}

}  // namespace

struct ReinforcementAgent::Implementation {
    explicit Implementation(ReinforcementAgentConfig value)
        : config(std::move(value)),
          options_by_character(make_all_options()),
          random_engine(config.random_seed) {
        validate_config();
        online = initialize_network(
            config, options_by_character, random_engine
        );
        target = online;
        first_moment = zero_vectors(online);
        second_moment = zero_vectors(online);
        character_best_episode_return.fill(
            -std::numeric_limits<float>::infinity()
        );
        rebuild_current_option_sets();
        if (
            config.resume &&
            !config.checkpoint_path.empty() &&
            std::filesystem::is_regular_file(config.checkpoint_path)
        ) {
            load(config.checkpoint_path);
            checkpoint_loaded = true;
        }
        if (config.trainable_lstm) {
            initialize_recurrent_backend();
            // Legacy replay entries contain only the old frozen LSTM output.
            // They remain preserved in the original checkpoint, but cannot be
            // used for sequence BPTT.
            replay.clear();
            replay_next = 0;
        }
    }

    void initialize_recurrent_backend() {
        std::vector<std::size_t> action_sizes;
        action_sizes.reserve(options_by_character.size());
        for (const auto& character_options : options_by_character) {
            action_sizes.push_back(character_options.size());
        }
        std::filesystem::path recurrent_checkpoint =
            config.recurrent_checkpoint_path;
        if (recurrent_checkpoint.empty()) {
            recurrent_checkpoint =
                config.checkpoint_path.string() + ".recurrent.pt";
        }
        recurrent_backend = std::make_unique<RecurrentPolicyBackend>(
            RecurrentPolicyBackendConfig{
                .service_script = config.recurrent_service_script,
                .checkpoint_path = recurrent_checkpoint,
                .resnet50_checkpoint_path =
                    config.resnet50_checkpoint_path,
                .action_sizes = std::move(action_sizes),
                .visual_size = config.temporal_visual_size,
                .projection_size = config.temporal_projection_size,
                .controller_size = config.temporal_controller_size,
                .temporal_hidden_size = config.temporal_hidden_size,
                .context_size =
                    config.input_size - config.temporal_hidden_size,
                .strategy_hidden_size = config.network_hidden_size,
                .sequence_capacity =
                    config.temporal_sequence_capacity,
                .character_embedding_size =
                    config.character_embedding_size,
                .opponent_embedding_size =
                    config.opponent_embedding_size,
                .opponent_count = kOpponentCharacterFeatureCount,
                .replay_capacity = config.replay_capacity,
                .minimum_replay_size = config.minimum_replay_size,
                .batch_size = config.batch_size,
                .training_updates_per_transition =
                    config.training_updates_per_transition,
                .target_update_interval =
                    config.target_update_interval,
                .initial_training_updates = training_updates,
                .gamma = config.gamma,
                .learning_rate = config.learning_rate,
                .resnet_learning_rate =
                    config.resnet_learning_rate,
                .gradient_clip = config.gradient_clip,
                .random_seed = config.random_seed,
                .device_id = config.temporal_device_id,
                .training_enabled = config.training_enabled,
                .checkpoint_writes_enabled =
                    config.checkpoint_writes_enabled,
                .resume = config.resume,
                .trainable_resnet = config.trainable_resnet,
                .resnet_batch_size = config.resnet_batch_size,
                .resnet_update_interval =
                    config.resnet_update_interval,
                .resnet_replay_images =
                    config.resnet_replay_images,
            }
        );
        const auto& recurrent_stats = recurrent_backend->stats();
        training_updates = recurrent_stats.training_updates;
        last_training_loss = recurrent_stats.last_loss;
    }

    const std::vector<OptionDefinition>& options() const {
        return options_by_character[character_index(config.character)];
    }

    void rebuild_current_option_sets() {
        primitive_options.clear();
        scripted_options.clear();
        const auto& current = options();
        for (std::size_t index = 0; index < current.size(); ++index) {
            (current[index].scripted
                 ? scripted_options
                 : primitive_options)
                .push_back(index);
        }
    }

    void validate_config() const {
        character_index(config.character);
        opponent_index(config.opponent_character);
        if (
            config.input_size == 0 ||
            config.network_hidden_size == 0 ||
            config.character_embedding_size == 0 ||
            config.opponent_embedding_size == 0 ||
            config.replay_capacity == 0 ||
            config.batch_size == 0 ||
            config.minimum_replay_size == 0 ||
            config.training_updates_per_transition == 0 ||
            config.target_update_interval == 0 ||
            config.gamma <= 0.0f ||
            config.gamma > 1.0f ||
            config.learning_rate <= 0.0f ||
            config.adam_beta1 <= 0.0f ||
            config.adam_beta1 >= 1.0f ||
            config.adam_beta2 <= 0.0f ||
            config.adam_beta2 >= 1.0f ||
            config.adam_epsilon <= 0.0f ||
            config.gradient_clip <= 0.0f ||
            config.epsilon_start < 0.0f ||
            config.epsilon_start > 1.0f ||
            config.epsilon_final < 0.0f ||
            config.epsilon_final > 1.0f ||
            config.epsilon_final > config.epsilon_start ||
            config.epsilon_decay_selections == 0 ||
            config.scripted_exploration_probability < 0.0f ||
            config.scripted_exploration_probability > 1.0f ||
            (
                config.trainable_resnet &&
                (
                    !config.trainable_lstm ||
                    config.resnet50_checkpoint_path.empty() ||
                    config.resnet_learning_rate <= 0.0f ||
                    config.resnet_batch_size == 0 ||
                    config.resnet_update_interval == 0 ||
                    config.resnet_replay_images == 0
                )
            ) ||
            (
                config.trainable_lstm &&
                (
                    config.temporal_visual_size == 0 ||
                    config.temporal_projection_size == 0 ||
                    config.temporal_controller_size == 0 ||
                    config.temporal_hidden_size == 0 ||
                    config.temporal_sequence_capacity == 0 ||
                    config.temporal_device_id < 0 ||
                    config.input_size <= config.temporal_hidden_size ||
                    config.recurrent_service_script.empty()
                )
            )
        ) {
            throw std::invalid_argument(
                "invalid reinforcement-agent configuration"
            );
        }
    }

    float epsilon(FighterCharacter character) const {
        const double progress = std::min(
            1.0,
            static_cast<double>(
                character_action_selections[character_index(character)]
            ) /
                static_cast<double>(
                    config.epsilon_decay_selections
                )
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
        const std::vector<float> values =
            recurrent_backend != nullptr
                ? recurrent_backend->q_values(
                      character_index(config.character),
                      opponent_index(config.opponent_character),
                      options().size()
                  )
                : forward(
                      online,
                      state,
                      config.character,
                      config.opponent_character
                  );
        std::uniform_real_distribution<float> probability(0.0f, 1.0f);
        std::size_t selected = 0;
        std::string source = "dqn_exploit";
        if (probability(random_engine) < epsilon(config.character)) {
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
        ++character_action_selections[
            character_index(config.character)
        ];
        return {selected, std::move(source)};
    }

    void append_replay(ReplayTransition transition) {
        if (replay.size() < config.replay_capacity) {
            replay.push_back(std::move(transition));
        } else {
            replay[replay_next] = std::move(transition);
            replay_next =
                (replay_next + 1U) % config.replay_capacity;
        }
    }

    static void accumulate_gradient(
        const SharedPolicyNetwork& network,
        const ReplayTransition& transition,
        float target_value,
        SharedPolicyVectors& gradient,
        double& loss
    ) {
        std::vector<float> augmented;
        std::vector<float> hidden;
        std::vector<float> preactivation;
        const std::vector<float> q = forward(
            network,
            transition.state,
            transition.character,
            transition.opponent,
            &augmented,
            &hidden,
            &preactivation
        );
        if (transition.action >= q.size()) {
            throw std::runtime_error(
                "replay action does not belong to character head"
            );
        }
        const float error = q[transition.action] - target_value;
        const float absolute_error = std::abs(error);
        loss += absolute_error <= 1.0f
            ? 0.5 * static_cast<double>(error) * error
            : static_cast<double>(absolute_error) - 0.5;
        const float output_gradient =
            std::clamp(error, -1.0f, 1.0f);
        const std::size_t player =
            character_index(transition.character);
        const std::size_t opponent =
            opponent_index(transition.opponent);
        const PolicyHead& head = network.heads[player];
        PolicyHead& head_gradient = gradient.heads[player];
        head_gradient.bias[transition.action] += output_gradient;
        const std::size_t head_offset =
            transition.action * network.hidden_size;
        const std::size_t augmented_size = augmented.size();
        for (
            std::size_t hidden_index = 0;
            hidden_index < network.hidden_size;
            ++hidden_index
        ) {
            head_gradient.weights[head_offset + hidden_index] +=
                output_gradient * hidden[hidden_index];
            if (preactivation[hidden_index] <= 0.0f) {
                continue;
            }
            const float hidden_gradient =
                output_gradient *
                head.weights[head_offset + hidden_index];
            gradient.shared_bias[hidden_index] += hidden_gradient;
            const std::size_t shared_offset =
                hidden_index * augmented_size;
            for (
                std::size_t input_index = 0;
                input_index < augmented_size;
                ++input_index
            ) {
                gradient.shared_weights[
                    shared_offset + input_index
                ] += hidden_gradient * augmented[input_index];
            }

            const std::size_t character_input_offset =
                network.state_size;
            const std::size_t character_embedding_offset =
                player * network.character_embedding_size;
            for (
                std::size_t index = 0;
                index < network.character_embedding_size;
                ++index
            ) {
                gradient.character_embeddings[
                    character_embedding_offset + index
                ] +=
                    hidden_gradient *
                    network.shared_weights[
                        shared_offset +
                        character_input_offset +
                        index
                    ];
            }

            const std::size_t opponent_input_offset =
                network.state_size +
                network.character_embedding_size;
            const std::size_t opponent_embedding_offset =
                opponent * network.opponent_embedding_size;
            for (
                std::size_t index = 0;
                index < network.opponent_embedding_size;
                ++index
            ) {
                gradient.opponent_embeddings[
                    opponent_embedding_offset + index
                ] +=
                    hidden_gradient *
                    network.shared_weights[
                        shared_offset +
                        opponent_input_offset +
                        index
                    ];
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
            parameter[index] -=
                config.learning_rate *
                static_cast<float>(
                    corrected_first /
                    (
                        std::sqrt(corrected_second) +
                        config.adam_epsilon
                    )
                );
        }
    }

    void adam_update(
        SharedPolicyNetwork& network,
        SharedPolicyVectors& first,
        SharedPolicyVectors& second,
        const SharedPolicyVectors& gradient,
        float inverse_batch
    ) {
        adam_update_vector(
            network.character_embeddings,
            first.character_embeddings,
            second.character_embeddings,
            gradient.character_embeddings,
            inverse_batch
        );
        adam_update_vector(
            network.opponent_embeddings,
            first.opponent_embeddings,
            second.opponent_embeddings,
            gradient.opponent_embeddings,
            inverse_batch
        );
        adam_update_vector(
            network.shared_weights,
            first.shared_weights,
            second.shared_weights,
            gradient.shared_weights,
            inverse_batch
        );
        adam_update_vector(
            network.shared_bias,
            first.shared_bias,
            second.shared_bias,
            gradient.shared_bias,
            inverse_batch
        );
        for (std::size_t index = 0; index < network.heads.size(); ++index) {
            adam_update_vector(
                network.heads[index].weights,
                first.heads[index].weights,
                second.heads[index].weights,
                gradient.heads[index].weights,
                inverse_batch
            );
            adam_update_vector(
                network.heads[index].bias,
                first.heads[index].bias,
                second.heads[index].bias,
                gradient.heads[index].bias,
                inverse_batch
            );
        }
    }

    std::vector<std::vector<std::size_t>> replay_matchup_buckets() const {
        std::vector<std::vector<std::size_t>> buckets(
            kControlledCharacterCount *
            kOpponentCharacterFeatureCount
        );
        for (std::size_t index = 0; index < replay.size(); ++index) {
            const ReplayTransition& transition = replay[index];
            const std::size_t matchup =
                character_index(transition.character) *
                    kOpponentCharacterFeatureCount +
                opponent_index(transition.opponent);
            buckets[matchup].push_back(index);
        }
        return buckets;
    }

    bool train_once() {
        const std::size_t minimum = std::max(
            config.minimum_replay_size, config.batch_size
        );
        if (!config.training_enabled || replay.size() < minimum) {
            return false;
        }
        const auto buckets = replay_matchup_buckets();
        std::vector<std::size_t> active_buckets;
        for (std::size_t index = 0; index < buckets.size(); ++index) {
            if (!buckets[index].empty()) {
                active_buckets.push_back(index);
            }
        }
        if (active_buckets.empty()) {
            return false;
        }

        SharedPolicyVectors gradient = zero_vectors(online);
        std::uniform_int_distribution<std::size_t> bucket_distribution(
            0, active_buckets.size() - 1U
        );
        double loss = 0.0;
        for (
            std::size_t sample = 0;
            sample < config.batch_size;
            ++sample
        ) {
            const auto& bucket =
                buckets[
                    active_buckets[
                        bucket_distribution(random_engine)
                    ]
                ];
            std::uniform_int_distribution<std::size_t>
                transition_distribution(0, bucket.size() - 1U);
            const ReplayTransition& transition =
                replay[
                    bucket[
                        transition_distribution(random_engine)
                    ]
                ];
            float target_value = transition.reward;
            if (!transition.terminal) {
                const std::vector<float> online_next = forward(
                    online,
                    transition.next_state,
                    transition.character,
                    transition.opponent
                );
                const std::size_t best_action =
                    static_cast<std::size_t>(std::distance(
                        online_next.begin(),
                        std::max_element(
                            online_next.begin(),
                            online_next.end()
                        )
                    ));
                const std::vector<float> target_next = forward(
                    target,
                    transition.next_state,
                    transition.character,
                    transition.opponent
                );
                target_value +=
                    std::pow(
                        config.gamma,
                        static_cast<float>(transition.duration)
                    ) *
                    target_next[best_action];
            }
            accumulate_gradient(
                online,
                transition,
                target_value,
                gradient,
                loss
            );
        }

        const float inverse_batch =
            1.0f / static_cast<float>(config.batch_size);
        adam_update(
            online,
            first_moment,
            second_moment,
            gradient,
            inverse_batch
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
        bool trained = false;
        if (recurrent_backend != nullptr) {
            const RecurrentPolicyCloseResult result =
                recurrent_backend->close_transition(
                    std::max<std::size_t>(1U, option_elapsed_steps),
                    option_reward,
                    terminal
                );
            trained = result.training_performed;
            training_updates = result.stats.training_updates;
            last_training_loss = result.stats.last_loss;
        } else if (config.training_enabled) {
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
                .character = config.character,
                .opponent = config.opponent_character,
            });
            for (
                std::size_t update = 0;
                update < config.training_updates_per_transition;
                ++update
            ) {
                trained = train_once() || trained;
            }
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
        if (!config.checkpoint_writes_enabled || path.empty()) {
            return;
        }
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path());
        }
        const std::filesystem::path temporary =
            path.string() + ".tmp";
        std::ofstream output(
            temporary,
            std::ios::binary | std::ios::trunc
        );
        if (!output) {
            throw std::runtime_error(
                "could not create shared reinforcement checkpoint: " +
                temporary.string()
            );
        }

        output.write(kCheckpointMagic.data(), kCheckpointMagic.size());
        write_value(output, kCheckpointVersion);
        write_value(
            output,
            static_cast<std::uint64_t>(config.input_size)
        );
        write_value(
            output,
            static_cast<std::uint64_t>(config.network_hidden_size)
        );
        write_value(
            output,
            static_cast<std::uint64_t>(
                config.character_embedding_size
            )
        );
        write_value(
            output,
            static_cast<std::uint64_t>(
                config.opponent_embedding_size
            )
        );
        write_value(
            output,
            static_cast<std::uint64_t>(kControlledCharacterCount)
        );
        write_value(
            output,
            static_cast<std::uint64_t>(
                kOpponentCharacterFeatureCount
            )
        );
        for (const auto& character_options : options_by_character) {
            write_value(
                output,
                static_cast<std::uint64_t>(
                    character_options.size()
                )
            );
            write_value(output, option_hash(character_options));
        }
        for (std::uint64_t value : character_action_selections) {
            write_value(output, value);
        }
        for (std::uint64_t value : character_episodes_completed) {
            write_value(output, value);
        }
        for (float value : character_best_episode_return) {
            write_value(output, value);
        }
        write_value(output, training_updates);
        write_value(output, current_episode_return);
        write_value(output, last_training_loss);
        write_network(output, online);
        write_network(output, target);
        write_vectors(output, first_moment);
        write_vectors(output, second_moment);

        const std::uint64_t replay_count = config.save_replay
            ? static_cast<std::uint64_t>(replay.size())
            : 0U;
        write_value(output, replay_count);
        write_value(
            output,
            static_cast<std::uint64_t>(replay_next)
        );
        for (
            std::size_t index = 0;
            index < static_cast<std::size_t>(replay_count);
            ++index
        ) {
            const ReplayTransition& transition = replay[index];
            write_vector(output, transition.state);
            write_vector(output, transition.next_state);
            write_value(output, transition.action);
            write_value(output, transition.duration);
            write_value(output, transition.reward);
            write_value(
                output,
                static_cast<std::uint8_t>(transition.terminal)
            );
            write_value(
                output,
                static_cast<std::uint8_t>(
                    character_index(transition.character)
                )
            );
            write_value(
                output,
                static_cast<std::uint8_t>(
                    opponent_index(transition.opponent)
                )
            );
        }
        write_value(output, kCheckpointTrailer);
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed to flush shared reinforcement checkpoint"
            );
        }
        output.close();
        std::filesystem::rename(temporary, path);
        if (recurrent_backend != nullptr) {
            const std::filesystem::path recurrent_path =
                path == config.checkpoint_path &&
                        !config.recurrent_checkpoint_path.empty()
                    ? config.recurrent_checkpoint_path
                    : std::filesystem::path(
                          path.string() + ".recurrent.pt"
                      );
            recurrent_backend->save(recurrent_path, false);
        }
    }

    void load(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "could not open shared reinforcement checkpoint: " +
                path.string()
            );
        }
        std::array<char, 8> magic{};
        input.read(magic.data(), magic.size());
        if (
            !input ||
            magic != kCheckpointMagic ||
            read_value<std::uint32_t>(input) != kCheckpointVersion
        ) {
            throw std::runtime_error(
                "invalid shared reinforcement checkpoint header"
            );
        }

        const auto state_size = read_value<std::uint64_t>(input);
        const auto hidden_size = read_value<std::uint64_t>(input);
        const auto character_embedding_size =
            read_value<std::uint64_t>(input);
        const auto opponent_embedding_size =
            read_value<std::uint64_t>(input);
        const auto character_count =
            read_value<std::uint64_t>(input);
        const auto opponent_count =
            read_value<std::uint64_t>(input);
        if (
            state_size != config.input_size ||
            hidden_size != config.network_hidden_size ||
            character_embedding_size !=
                config.character_embedding_size ||
            opponent_embedding_size !=
                config.opponent_embedding_size ||
            character_count != kControlledCharacterCount ||
            opponent_count != kOpponentCharacterFeatureCount
        ) {
            throw std::runtime_error(
                "shared reinforcement checkpoint architecture mismatch"
            );
        }
        for (const auto& character_options : options_by_character) {
            const auto action_size =
                read_value<std::uint64_t>(input);
            const auto saved_hash =
                read_value<std::uint64_t>(input);
            if (
                action_size != character_options.size() ||
                saved_hash != option_hash(character_options)
            ) {
                throw std::runtime_error(
                    "shared reinforcement checkpoint options mismatch"
                );
            }
        }
        for (std::uint64_t& value : character_action_selections) {
            value = read_value<std::uint64_t>(input);
        }
        for (std::uint64_t& value : character_episodes_completed) {
            value = read_value<std::uint64_t>(input);
        }
        for (float& value : character_best_episode_return) {
            value = read_value<float>(input);
        }
        training_updates = read_value<std::uint64_t>(input);
        current_episode_return = read_value<float>(input);
        last_training_loss = read_value<float>(input);
        read_network(input, online);
        read_network(input, target);
        read_vectors(input, first_moment);
        read_vectors(input, second_moment);

        const std::uint64_t replay_count =
            read_value<std::uint64_t>(input);
        replay_next = static_cast<std::size_t>(
            read_value<std::uint64_t>(input)
        );
        if (replay_count > config.replay_capacity) {
            throw std::runtime_error(
                "shared reinforcement checkpoint replay is too large"
            );
        }
        replay.clear();
        replay.reserve(static_cast<std::size_t>(replay_count));
        for (std::uint64_t index = 0; index < replay_count; ++index) {
            ReplayTransition transition;
            transition.state =
                read_vector(input, config.input_size);
            transition.next_state =
                read_vector(input, config.input_size);
            transition.action = read_value<std::uint32_t>(input);
            transition.duration = read_value<std::uint32_t>(input);
            transition.reward = read_value<float>(input);
            transition.terminal =
                read_value<std::uint8_t>(input) != 0;
            const std::uint8_t player =
                read_value<std::uint8_t>(input);
            const std::uint8_t opponent =
                read_value<std::uint8_t>(input);
            if (
                player >= kControlledCharacterCount ||
                opponent >= kOpponentCharacterFeatureCount
            ) {
                throw std::runtime_error(
                    "invalid shared replay character identity"
                );
            }
            transition.character = kControlledCharacters[player];
            transition.opponent =
                static_cast<OpponentCharacter>(opponent);
            if (
                transition.action >=
                    options_by_character[player].size() ||
                transition.duration == 0 ||
                !std::isfinite(transition.reward)
            ) {
                throw std::runtime_error(
                    "invalid shared replay transition"
                );
            }
            replay.push_back(std::move(transition));
        }
        if (replay.size() < config.replay_capacity) {
            replay_next = 0;
        } else if (replay_next >= replay.size()) {
            throw std::runtime_error(
                "invalid shared replay cursor"
            );
        }
        if (
            read_value<std::uint64_t>(input) !=
            kCheckpointTrailer
        ) {
            throw std::runtime_error(
                "shared reinforcement checkpoint trailer mismatch"
            );
        }
    }

    std::size_t shared_replay_matchup_count() const {
        const auto buckets = replay_matchup_buckets();
        return static_cast<std::size_t>(std::count_if(
            buckets.begin(),
            buckets.end(),
            [](const auto& bucket) { return !bucket.empty(); }
        ));
    }

    ReinforcementAgentConfig config;
    CharacterOptions options_by_character;
    std::vector<std::size_t> primitive_options;
    std::vector<std::size_t> scripted_options;
    std::mt19937 random_engine;
    SharedPolicyNetwork online;
    SharedPolicyNetwork target;
    SharedPolicyVectors first_moment;
    SharedPolicyVectors second_moment;
    std::unique_ptr<RecurrentPolicyBackend> recurrent_backend;
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
    std::array<
        std::uint64_t,
        kControlledCharacterCount
    > character_action_selections{};
    std::array<
        std::uint64_t,
        kControlledCharacterCount
    > character_episodes_completed{};
    std::array<
        float,
        kControlledCharacterCount
    > character_best_episode_return{};
    std::uint64_t training_updates = 0;
    std::uint64_t current_episode_id = 0;
    float current_episode_return = 0.0f;
    float last_training_loss = 0.0f;
    bool checkpoint_loaded = false;
};

ReinforcementAgent::ReinforcementAgent(ReinforcementAgentConfig config)
    : implementation_(
          std::make_unique<Implementation>(std::move(config))
      ) {}

ReinforcementAgent::~ReinforcementAgent() = default;
ReinforcementAgent::ReinforcementAgent(
    ReinforcementAgent&&
) noexcept = default;
ReinforcementAgent& ReinforcementAgent::operator=(
    ReinforcementAgent&&
) noexcept = default;

ReinforcementDecision ReinforcementAgent::step(
    const ReinforcementStepInput& input
) {
    auto& state = *implementation_;
    if (!input.state.empty()) {
        if (input.state.size() != state.config.input_size) {
            throw std::invalid_argument(
                "reinforcement step state size mismatch"
            );
        }
        require_finite(input.state, "reinforcement state");
    }
    if (!std::isfinite(input.reward)) {
        throw std::invalid_argument(
            "reinforcement reward is not finite"
        );
    }
    if (
        state.recurrent_backend != nullptr &&
        !input.state.empty()
    ) {
        if (
            (
                !state.config.trainable_resnet &&
                input.temporal_visual_feature.size() !=
                    state.config.temporal_visual_size
            ) ||
            (
                state.config.trainable_resnet &&
                input.temporal_jpeg.empty()
            ) ||
            input.temporal_controller_features.size() !=
                state.config.temporal_controller_size
        ) {
            throw std::invalid_argument(
                "trainable LSTM requires the current visual/controller "
                "observation"
            );
        }
        const std::span<const float> context =
            input.state.subspan(state.config.temporal_hidden_size);
        state.recurrent_backend->observe(
            input.temporal_visual_feature,
            input.temporal_jpeg,
            input.frame_id,
            input.temporal_controller_features,
            context,
            input.episode_started
        );
    }

    ReinforcementDecision decision;
    decision.reward_consumed = input.reward;
    if (
        input.episode_started ||
        (
            input.episode_id != 0 &&
            input.episode_id != state.current_episode_id
        )
    ) {
        state.current_episode_id = input.episode_id;
        state.current_episode_return = 0.0f;
    }
    state.current_episode_return += input.reward;
    if (state.active_option.has_value()) {
        state.option_reward +=
            state.option_reward_discount * input.reward;
        state.option_reward_discount *= state.config.gamma;
    }

    bool trained = false;
    const auto& options = state.options();
    const bool option_complete =
        state.active_option.has_value() &&
        state.active_step >=
            options[*state.active_option].states.size();
    if (input.episode_ended || option_complete) {
        trained = state.close_active_transition(
            input.state, input.episode_ended
        );
    }

    if (input.episode_ended) {
        const std::size_t player =
            character_index(state.config.character);
        ++state.character_episodes_completed[player];
        const bool new_best =
            state.current_episode_return >
            state.character_best_episode_return[player];
        if (new_best) {
            state.character_best_episode_return[player] =
                state.current_episode_return;
        }
        if (
            state.config.checkpoint_writes_enabled &&
            !state.config.checkpoint_path.empty()
        ) {
            state.write_checkpoint(state.config.checkpoint_path);
            decision.checkpoint_saved = true;
        }
        if (
            state.config.checkpoint_writes_enabled &&
            new_best &&
            !state.config.best_checkpoint_path.empty()
        ) {
            state.write_checkpoint(
                state.config.best_checkpoint_path
            );
        }
        if (state.recurrent_backend != nullptr) {
            state.recurrent_backend->reset();
        }
    }

    if (
        input.action_allowed &&
        input.state.size() == state.config.input_size
    ) {
        if (!state.active_option.has_value()) {
            float q = 0.0f;
            auto [selected, source] =
                state.select_action(input.state, q);
            state.active_option = selected;
            state.active_step = 0;
            state.option_start_state.assign(
                input.state.begin(), input.state.end()
            );
            state.option_reward = 0.0f;
            state.option_reward_discount = 1.0f;
            state.option_elapsed_steps = 0;
            state.active_selection_source = std::move(source);
            state.active_selected_q = q;
            if (state.recurrent_backend != nullptr) {
                state.recurrent_backend->begin_transition(
                    selected,
                    character_index(state.config.character),
                    opponent_index(state.config.opponent_character)
                );
            }
        }
        const std::size_t selected = *state.active_option;
        const OptionDefinition& option = options[selected];
        decision.commands = option.states[state.active_step];
        if (state.config.facing == FighterFacing::left) {
            for (int& command : decision.commands) {
                if (command == kLeft) {
                    command = kRight;
                } else if (command == kRight) {
                    command = kLeft;
                }
            }
        }
        decision.option_name = option.name;
        decision.selection_source =
            state.active_selection_source;
        decision.option_index = selected;
        decision.option_step = state.active_step + 1U;
        decision.option_steps = option.states.size();
        decision.selected_q = state.active_selected_q;
        ++state.active_step;
        ++state.option_elapsed_steps;
    }

    decision.epsilon = state.epsilon(state.config.character);
    decision.episode_return = state.current_episode_return;
    if (state.recurrent_backend != nullptr) {
        const auto& recurrent = state.recurrent_backend->stats();
        decision.training_loss = recurrent.last_loss;
        decision.replay_size = recurrent.replay_size;
        decision.training_updates = recurrent.training_updates;
    } else {
        decision.training_loss = state.last_training_loss;
        decision.replay_size = state.replay.size();
        decision.training_updates = state.training_updates;
    }
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
    if (state.recurrent_backend != nullptr) {
        state.recurrent_backend->reset();
    }
    if (
        save_checkpoint &&
        state.config.checkpoint_writes_enabled &&
        !state.config.checkpoint_path.empty()
    ) {
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
    const std::size_t player =
        character_index(state.config.character);
    const RecurrentPolicyBackendStats* recurrent =
        state.recurrent_backend != nullptr
            ? &state.recurrent_backend->stats()
            : nullptr;
    return {
        .action_selections =
            state.character_action_selections[player],
        .training_updates =
            recurrent != nullptr
                ? recurrent->training_updates
                : state.training_updates,
        .episodes_completed =
            state.character_episodes_completed[player],
        .replay_size =
            recurrent != nullptr
                ? recurrent->replay_size
                : state.replay.size(),
        .epsilon = state.epsilon(state.config.character),
        .last_training_loss =
            recurrent != nullptr
                ? recurrent->last_loss
                : state.last_training_loss,
        .current_episode_return = state.current_episode_return,
        .best_episode_return =
            state.character_best_episode_return[player],
        .shared_parameter_checksum =
            recurrent != nullptr
                ? recurrent->parameter_checksum
                : shared_parameter_checksum(state.online),
        .shared_replay_matchups =
            recurrent != nullptr
                ? recurrent->matchup_count
                : state.shared_replay_matchup_count(),
        .checkpoint_loaded = state.checkpoint_loaded,
        .trainable_lstm = recurrent != nullptr,
        .recurrent_checkpoint_loaded =
            recurrent != nullptr && recurrent->checkpoint_loaded,
        .trainable_resnet =
            recurrent != nullptr && recurrent->trainable_resnet,
        .resnet_checkpoint_loaded =
            recurrent != nullptr && recurrent->resnet_checkpoint_loaded,
        .resnet_training_updates =
            recurrent != nullptr
                ? recurrent->resnet_training_updates
                : 0,
        .resnet_parameter_checksum =
            recurrent != nullptr
                ? recurrent->resnet_parameter_checksum
                : 0.0,
        .training_device =
            recurrent != nullptr ? recurrent->device : "cpu",
    };
}

std::vector<std::string> ReinforcementAgent::option_names() const {
    std::vector<std::string> names;
    const auto& options = implementation_->options();
    names.reserve(options.size());
    for (const auto& option : options) {
        names.push_back(option.name);
    }
    return names;
}

const ReinforcementAgentConfig& ReinforcementAgent::config() const {
    return implementation_->config;
}

}  // namespace gameagent
