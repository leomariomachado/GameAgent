#include "GameProfile.h"

#include <optional>
#include <stdexcept>

namespace gameagent {
namespace {

const GameProfile kUltimateMortalKombat3{
    .id = GameProfileId::ultimate_mortal_kombat_3,
    .slug = "ultimate_mortal_kombat_3",
    .display_name = "Ultimate Mortal Kombat 3",
    .action_intents = {},
    .combat_state_adapter = true,
    .reinforcement_runtime_ready = true,
};

// These are controller intents, not emulator key bindings.  The Mega Man
// client adapter will map each existing command code to its configured pad
// button while retaining the V2 action envelope and acknowledgements.
const GameProfile kMegaManX2{
    .id = GameProfileId::megaman_x2,
    .slug = "megaman_x2",
    .display_name = "Mega Man X2",
    .action_intents = {
        {"idle", "Parado", {}},
        {"move_left", "Mover para esquerda", {8}},
        {"move_right", "Mover para direita", {9}},
        {"jump", "Pular", {6}},
        {"shoot", "Atirar", {3}},
        {"dash", "Dash", {5}},
        {"jump_left", "Pular para esquerda", {6, 8}},
        {"jump_right", "Pular para direita", {6, 9}},
        {"jump_shoot", "Pular e atirar", {6, 3}},
        {"dash_shoot", "Dash e atirar", {5, 3}},
    },
    .combat_state_adapter = false,
    .reinforcement_runtime_ready = false,
};

}  // namespace

std::string_view game_profile_slug(GameProfileId profile) {
    return game_profile(profile).slug;
}

std::string_view game_profile_name(GameProfileId profile) {
    return game_profile(profile).display_name;
}

std::vector<std::string> supported_game_profiles() {
    return {"ultimate_mortal_kombat_3", "megaman_x2"};
}

std::optional<GameProfileId> parse_game_profile(std::string_view value) {
    if (value == "ultimate_mortal_kombat_3" || value == "umk3") {
        return GameProfileId::ultimate_mortal_kombat_3;
    }
    if (value == "megaman_x2" || value == "mega_man_x2") {
        return GameProfileId::megaman_x2;
    }
    return std::nullopt;
}

const GameProfile& game_profile(GameProfileId profile) {
    switch (profile) {
        case GameProfileId::ultimate_mortal_kombat_3:
            return kUltimateMortalKombat3;
        case GameProfileId::megaman_x2:
            return kMegaManX2;
    }
    throw std::invalid_argument("unknown game profile");
}

}  // namespace gameagent
