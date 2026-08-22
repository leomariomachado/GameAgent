#pragma once

#include <string>
#include <string_view>
#include <optional>
#include <vector>

namespace gameagent {

// A game profile owns the game-specific contract: available controller
// intents, state adapter and reward model.  Shared transport, perception and
// temporal infrastructure deliberately remain outside this boundary.
enum class GameProfileId {
    ultimate_mortal_kombat_3,
    megaman_x2,
};

struct GameActionIntent {
    std::string id;
    std::string label;
    std::vector<int> commands;
};

struct GameProfile {
    GameProfileId id;
    std::string_view slug;
    std::string_view display_name;
    std::vector<GameActionIntent> action_intents;
    bool combat_state_adapter = false;
    bool reinforcement_runtime_ready = false;
};

std::string_view game_profile_slug(GameProfileId profile);
std::string_view game_profile_name(GameProfileId profile);
std::vector<std::string> supported_game_profiles();
std::optional<GameProfileId> parse_game_profile(std::string_view value);
const GameProfile& game_profile(GameProfileId profile);

}  // namespace gameagent
