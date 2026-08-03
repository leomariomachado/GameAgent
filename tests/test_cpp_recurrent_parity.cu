#include "RecurrentPolicyBackend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <typename T>
void read_values(std::istream& input, std::vector<T>& values) {
    for (T& value : values) {
        if (!(input >> value)) {
            throw std::runtime_error("truncated recurrent parity fixture");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument("usage: test CHECKPOINT FIXTURE");
        }
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
            device_count == 0) {
            std::cout << "[SKIP] CUDA device unavailable\n";
            return 77;
        }
        std::ifstream input(argv[2]);
        if (!input) {
            throw std::runtime_error("could not open parity fixture");
        }
        gameagent::RecurrentPolicyBackendConfig config;
        std::size_t character_count = 0;
        input >> config.visual_size
              >> config.projection_size
              >> config.controller_size
              >> config.temporal_hidden_size
              >> config.context_size
              >> config.strategy_hidden_size
              >> config.sequence_capacity
              >> config.character_embedding_size
              >> config.opponent_embedding_size
              >> config.opponent_count
              >> character_count;
        config.action_sizes.resize(character_count);
        read_values(input, config.action_sizes);
        std::size_t character = 0;
        std::size_t opponent = 0;
        std::size_t steps = 0;
        input >> character >> opponent >> steps;
        config.backend = "cpp";
        config.checkpoint_path = argv[1];
        config.replay_capacity = 16;
        config.minimum_replay_size = 4;
        config.batch_size = 4;
        config.training_enabled = false;
        config.checkpoint_writes_enabled = false;
        config.resume = true;
        config.trainable_resnet = false;
        gameagent::RecurrentPolicyBackend backend(config);
        for (std::size_t step = 0; step < steps; ++step) {
            std::vector<float> visual(config.visual_size);
            std::vector<float> controller(config.controller_size);
            std::vector<float> context(config.context_size);
            read_values(input, visual);
            read_values(input, controller);
            read_values(input, context);
            backend.observe(
                visual, {}, step + 1, controller, context, step == 0
            );
        }
        std::vector<float> expected(config.action_sizes[character]);
        read_values(input, expected);
        const std::vector<float> actual = backend.q_values(
            character, opponent, expected.size()
        );
        float maximum_difference = 0.0f;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            maximum_difference = std::max(
                maximum_difference,
                std::abs(expected[index] - actual[index])
            );
        }
        if (maximum_difference > 3.0e-4f) {
            throw std::runtime_error(
                "PyTorch/C++ Q-value parity exceeded tolerance: " +
                std::to_string(maximum_difference)
            );
        }
        std::cout
            << "[INFO] PyTorch/C++ maximum_abs="
            << maximum_difference << '\n'
            << "[PASS] Native recurrent forward parity passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
