#include "HealthBarDetector.h"

#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(float actual, float expected, float tolerance, const char* message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void draw_bar(
    base::data::image::Image& image,
    int x_begin,
    int x_end,
    float health,
    bool blue_from_left
) {
    const int columns = x_end - x_begin;
    const int blue_columns = static_cast<int>(std::lround(health * columns));
    for (int y = 39; y < 54; ++y) {
        for (int x = x_begin; x < x_end; ++x) {
            const int relative = x - x_begin;
            const bool blue = blue_from_left
                ? relative < blue_columns
                : relative >= columns - blue_columns;
            image.at(y, x, 0) = blue ? 8U : 120U;
            image.at(y, x, 1) = blue ? 18U : 8U;
            image.at(y, x, 2) = blue ? 230U : 8U;
        }
    }
}

gameagent::BufferedFrame make_frame(
    std::uint64_t frame_id,
    std::int64_t timestamp_ms,
    float left_health,
    float right_health,
    bool valid = true
) {
    base::data::image::Image image(512, 448, 3);
    std::fill(image.bytes().begin(), image.bytes().end(), 0U);
    if (valid) {
        draw_bar(image, 31, 225, left_health, true);
        draw_bar(image, 287, 481, right_health, false);
    }
    return {
        .frame_id = frame_id,
        .client_timestamp_ms = timestamp_ms,
        .received_timestamp_ms = timestamp_ms + 5,
        .jpeg_size = 0,
        .image = std::move(image),
    };
}

base::data::image::Image load_jpeg(const std::filesystem::path& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* decoded = stbi_load(
        path.c_str(), &width, &height, &channels, 3
    );
    if (decoded == nullptr) {
        throw std::runtime_error("could not decode real health sample");
    }
    struct Guard {
        unsigned char* value;
        ~Guard() { stbi_image_free(value); }
    } guard{decoded};
    base::data::image::Image image(width, height, 3);
    std::memcpy(image.data(), decoded, image.size_bytes());
    return image;
}

}  // namespace

int main() {
    try {
        gameagent::HealthBarDetector detector;
        auto measured = detector.measure(make_frame(1, 1000, 0.28f, 0.72f).image);
        require(measured.hud_valid, "synthetic fight HUD was not detected");
        require_near(measured.left_health, 0.28f, 0.015f, "left health is inaccurate");
        require_near(measured.right_health, 0.72f, 0.015f, "right health is inaccurate");
        require(
            !detector.measure(make_frame(2, 1050, 0.0f, 0.0f, false).image).hud_valid,
            "blank image was accepted as a fight HUD"
        );
        if (const char* sample_directory =
                std::getenv("GAMEAGENT_HEALTH_SAMPLE_DIR");
            sample_directory != nullptr && sample_directory[0] != '\0') {
            const std::filesystem::path root(sample_directory);
            const auto frame107 = detector.measure(
                load_jpeg(root / "frame_107.jpg")
            );
            const auto frame298 = detector.measure(
                load_jpeg(root / "frame_298.jpg")
            );
            const auto frame834 = detector.measure(
                load_jpeg(root / "frame_834.jpg")
            );
            const auto frame1256 = detector.measure(
                load_jpeg(root / "frame_1256.jpg")
            );
            require(frame107.hud_valid, "real low-health HUD was rejected");
            require_near(frame107.left_health, 0.278f, 0.02f, "real frame 107 left health is inaccurate");
            require_near(frame107.right_health, 0.990f, 0.02f, "real frame 107 right health is inaccurate");
            require(frame298.hud_valid, "real damaged HUD was rejected");
            require_near(frame298.left_health, 0.835f, 0.02f, "real frame 298 left health is inaccurate");
            require_near(frame298.right_health, 0.722f, 0.02f, "real frame 298 right health is inaccurate");
            require(
                frame834.hud_valid &&
                    frame834.left_health > 0.97f &&
                    frame834.right_health > 0.97f,
                "real round-start bars were not measured as full"
            );
            require(!frame1256.hud_valid, "real non-fight screen was accepted");
            std::cout << "[INFO] real capture samples validated\n";
        }

        gameagent::FightStateTracker tracker({
            .player_side = gameagent::PlayerSide::left,
            .valid_frames_to_acquire = 3,
            .invalid_frames_to_end = 2,
            .stable_intro_frames = 3,
            .terminal_frames = 2,
            .round_intro_grace_ms = 100,
            .maximum_action_frame_age_ms = 250,
        });
        tracker.observe(make_frame(10, 1000, 0.80f, 0.70f));
        tracker.observe(make_frame(11, 1050, 0.80f, 0.70f));
        auto active = tracker.observe(make_frame(12, 1100, 0.80f, 0.70f));
        require(
            active.phase == gameagent::FightPhase::active_fight &&
                active.action_allowed,
            "stable mid-round HUD did not enable control"
        );

        tracker.observe(make_frame(13, 1150, 0.80f, 0.60f));
        auto reward = tracker.consume(1160);
        require_near(reward.damage_dealt, 0.10f, 0.02f, "opponent damage is inaccurate");
        require_near(reward.reward, 1.5f, 0.2f, "damage reward is inaccurate");

        tracker.observe(make_frame(14, 1200, 0.70f, 0.60f));
        reward = tracker.consume(1210);
        require_near(reward.damage_taken, 0.10f, 0.02f, "player damage is inaccurate");
        require_near(reward.reward, -1.1f, 0.2f, "received damage penalty is inaccurate");

        auto refill = tracker.observe(make_frame(15, 1300, 1.0f, 1.0f));
        require(
            refill.phase == gameagent::FightPhase::round_intro &&
                !refill.action_allowed,
            "health refill did not close control during round intro"
        );
        auto boundary = tracker.consume(1310);
        require(boundary.episode_ended, "health refill did not end the prior episode");
        require_near(boundary.reward, 0.0f, 0.001f, "refill generated a reward");

        tracker.observe(make_frame(16, 1350, 1.0f, 1.0f));
        tracker.observe(make_frame(17, 1400, 1.0f, 1.0f));
        active = tracker.observe(make_frame(18, 1450, 1.0f, 1.0f));
        require(active.action_allowed, "stable round intro did not reopen control");

        tracker.observe(make_frame(19, 1500, 1.0f, 0.0f));
        auto terminal = tracker.observe(make_frame(20, 1550, 1.0f, 0.0f));
        require(
            terminal.phase == gameagent::FightPhase::round_end &&
                !terminal.action_allowed,
            "zero opponent health did not end the round"
        );
        terminal = tracker.consume(1560);
        require(terminal.terminal && terminal.episode_ended, "terminal flags were lost");
        require_near(terminal.terminal_reward, 5.0f, 0.001f, "round win bonus is inaccurate");
        require_near(terminal.reward, 20.0f, 0.2f, "round win did not include damage and terminal bonus");

        tracker.observe(make_frame(21, 1600, 0.0f, 0.0f, false));
        const auto invalid = tracker.snapshot(1610);
        require(!invalid.action_allowed, "invalid HUD allowed an action");

        std::cout
            << "[INFO] bar_columns=" << measured.bar_columns
            << " left=" << measured.left_health
            << " right=" << measured.right_health
            << " confidence=" << measured.confidence << '\n'
            << "[PASS] health bars, fight gating and reward test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
