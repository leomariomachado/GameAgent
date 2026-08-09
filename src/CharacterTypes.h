#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gameagent {

enum class FighterFacing {
    right,
    left,
};

enum class FighterCharacter {
    subzero,
    scorpion,
    noob_saibot,
    sindel,
};

enum class OpponentCharacter : std::uint8_t {
    unknown = 0,
    cyrax,
    ermac,
    jade,
    jax,
    kabal,
    kano,
    kitana,
    kung_lao,
    liu_kang,
    mileena,
    motaro,
    nightwolf,
    noob_saibot,
    rain,
    reptile,
    scorpion,
    sektor,
    shang_tsung,
    shao_kahn,
    sheeva,
    sindel,
    smoke,
    human_smoke,
    sonya,
    stryker,
    subzero,
    classic_subzero,
    count,
};

inline constexpr std::size_t kOpponentCharacterFeatureCount =
    static_cast<std::size_t>(OpponentCharacter::count);

std::optional<FighterCharacter> parse_fighter_character(
    std::string_view value
);
std::string_view fighter_character_slug(FighterCharacter character);
std::string_view fighter_character_name(FighterCharacter character);
std::vector<std::string> supported_fighter_characters();
std::optional<OpponentCharacter> parse_opponent_character(
    std::string_view value
);
std::string_view opponent_character_slug(OpponentCharacter character);
std::string_view opponent_character_name(OpponentCharacter character);
std::vector<std::string> supported_opponent_characters();

}  // namespace gameagent
