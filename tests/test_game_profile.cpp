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
    require(profile.action_intents.size() == 10U,
            "Mega Man action catalogue is incomplete");
    require(profile.action_intents.at(7).id == "jump_right" &&
                profile.action_intents.at(7).commands == std::vector<int>({6, 9}),
            "Mega Man jump-right intent is inconsistent");
    require(!parse_game_profile("unknown").has_value(),
            "unknown game profile was accepted");
    std::cout << "[PASS] game profiles are isolated and UMK3 remains ready\n";
}
