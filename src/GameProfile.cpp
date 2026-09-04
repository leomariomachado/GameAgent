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

// These programs use the controller codes already established by protocol V2:
// X=1, B=2, A=3, Up=6, Down=7, Left=8 and Right=9.  B is the shot button,
// A is jump and X is dash in the Mega Man X client layout.  Each program has
// an explicit final release, preventing a charged shot or wall movement from
// remaining held after its action completes.
const GameProfile kMegaManX2{
    .id = GameProfileId::megaman_x2,
    // `megaman_x` is the public, game-family identifier.  X1/X2/X3 are
    // accepted as input aliases, but a session acknowledgement must always
    // return this canonical value so clients do not reject the configuration
    // because they selected a different entry in the same family.
    .slug = "megaman_x",
    .display_name = "Mega Man X",
    .action_intents = {
        {"idle", "Parado", {{}}, 0, 0, 100},
        {"move_left", "Mover para esquerda", {{8}}, 120, 40, 100},
        {"move_right", "Mover para direita", {{9}}, 120, 40, 100},
        {"shoot", "Tiro", {{2}}, 75, 40, 100},
        {"charge_shot", "Tiro concentrado", {{2}}, 600, 40, 100},
        {"super_charge_shot", "Tiro super concentrado", {{2}}, 1200, 40, 100},
        {"double_charge_shot", "Tiro duplo concentrado (X2)", {{2}}, 1800, 40, 100},
        {"jump", "Pulo", {{3}}, 100, 40, 100},
        {"air_dash", "Dash aéreo", {{3}, {1}}, 90, 35, 100},
        {"long_jump", "Pulo longo (B+A)", {{2, 3}}, 350, 40, 100},
        {"wall_climb_right", "Escalar parede à direita", {{9, 2}, {}, {9, 2}, {}, {9, 2}}, 90, 35, 100},
        {"wall_climb_left", "Escalar parede à esquerda", {{8, 2}, {}, {8, 2}, {}, {8, 2}}, 90, 35, 100},
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
    return {"ultimate_mortal_kombat_3", "megaman_x"};
}

std::optional<GameProfileId> parse_game_profile(std::string_view value) {
    if (value == "ultimate_mortal_kombat_3" || value == "umk3") {
        return GameProfileId::ultimate_mortal_kombat_3;
    }
    if (
        value == "megaman_x" || value == "mega_man_x" ||
        value == "megaman_x1" || value == "megaman_x2" ||
        value == "megaman_x3" || value == "mega_man_x2"
    ) {
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
