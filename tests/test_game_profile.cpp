#include "GameProfile.h"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
}

int main() {
    using namespace gameagent;
    const auto umk3 = parse_game_profile("umk3");
    require(umk3.has_value(), "UMK3 alias was not recognized");
    require(
        game_profile(*umk3).reinforcement_runtime_ready,
        "UMK3 runtime compatibility changed"
    );

    const auto megaman = parse_game_profile("megaman_x2");
    require(megaman.has_value(), "Mega Man X2 was not recognized");
    const auto& profile = game_profile(*megaman);
    require(!profile.reinforcement_runtime_ready,
            "Mega Man was marked trainable before its state adapter exists");
    require(profile.action_intents.size() == 12U,
            "Mega Man action catalogue is incomplete");
    require(profile.action_intents.at(4).id == "charge_shot" &&
                profile.action_intents.at(4).states ==
                    std::vector<std::vector<int>>({{2}}) &&
                profile.action_intents.at(4).input_hold_ms == 600U,
            "Mega Man charge-shot intent is inconsistent");
    require(profile.action_intents.at(10).id == "wall_climb_right" &&
                profile.action_intents.at(10).states.size() == 5U,
            "Mega Man wall-climb intent is inconsistent");
    require(!parse_game_profile("unknown").has_value(),
            "unknown game profile was accepted");
    std::cout << "[PASS] game profiles are isolated and UMK3 remains ready\n";
}
