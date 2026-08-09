#pragma once

#include "RecurrentPolicyBackend.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace gameagent {

// In-process CUDA implementation of the recurrent clipped twin Double-DQN.
// The implementation deliberately lives in GameAgent: it consumes BaseAI's
// Tensor/Stream/AdamW/checkpoint primitives without adding a recurrent layer
// to the BaseAI public API while the design is still being validated.
class CppRecurrentPolicyBackend {
public:
    explicit CppRecurrentPolicyBackend(RecurrentPolicyBackendConfig config);
    ~CppRecurrentPolicyBackend();

    CppRecurrentPolicyBackend(const CppRecurrentPolicyBackend&) = delete;
    CppRecurrentPolicyBackend& operator=(
        const CppRecurrentPolicyBackend&
    ) = delete;

    void observe(
        std::span<const float> visual,
        std::span<const std::uint8_t> jpeg,
        std::uint64_t frame_id,
        std::span<const float> controller,
        std::span<const float> context,
        bool episode_started
    );
    std::vector<float> q_values(
        std::size_t character,
        std::size_t opponent,
        std::size_t action_count
    );
    void begin_transition(
        std::size_t action,
        std::size_t character,
        std::size_t opponent,
        float selected_q
    );
    RecurrentPolicyCloseResult close_transition(
        std::size_t duration,
        float reward,
        bool terminal,
        bool successful_episode,
        std::span<const std::uint8_t> next_action_mask
    );
    RecurrentPolicyCloseResult finalize_episode(
        float terminal_reward,
        bool successful_episode,
        std::span<const std::uint8_t> next_action_mask
    );
    void reset();
    void set_training_enabled(bool enabled);
    void save(
        const std::filesystem::path& path = {},
        bool include_replay = false
    );

    [[nodiscard]] const RecurrentPolicyBackendStats& stats() const;
    [[nodiscard]] const RecurrentPolicyBackendConfig& config() const;

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace gameagent
