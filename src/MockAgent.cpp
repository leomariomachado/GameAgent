#include "MockAgent.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <iterator>
#include <stdexcept>

namespace gameagent {
namespace {

constexpr int kYHighPunch = 0;
constexpr int kXHighKick = 1;
constexpr int kBLowPunch = 2;
constexpr int kALowKick = 3;
constexpr int kBlock = 4;
constexpr int kUp = 6;
constexpr int kDown = 7;
constexpr int kLeft = 8;
constexpr int kRight = 9;
constexpr std::size_t kDisableCloudCooldownMoves = 2;

bool is_disable_cloud_move(std::string_view name) {
    return name.rfind("Nuvem de Desativação", 0) == 0;
}

struct OpponentCharacterRecord {
    OpponentCharacter character;
    std::string_view slug;
    std::string_view name;
};

constexpr std::array<
    OpponentCharacterRecord,
    kOpponentCharacterFeatureCount
> kOpponentCharacters{{
    {OpponentCharacter::unknown, "unknown", "Unknown"},
    {OpponentCharacter::cyrax, "cyrax", "Cyrax"},
    {OpponentCharacter::ermac, "ermac", "Ermac"},
    {OpponentCharacter::jade, "jade", "Jade"},
    {OpponentCharacter::jax, "jax", "Jax"},
    {OpponentCharacter::kabal, "kabal", "Kabal"},
    {OpponentCharacter::kano, "kano", "Kano"},
    {OpponentCharacter::kitana, "kitana", "Kitana"},
    {OpponentCharacter::kung_lao, "kung_lao", "Kung Lao"},
    {OpponentCharacter::liu_kang, "liu_kang", "Liu Kang"},
    {OpponentCharacter::mileena, "mileena", "Mileena"},
    {OpponentCharacter::motaro, "motaro", "Motaro"},
    {OpponentCharacter::nightwolf, "nightwolf", "Nightwolf"},
    {OpponentCharacter::noob_saibot, "noob_saibot", "Noob Saibot"},
    {OpponentCharacter::rain, "rain", "Rain"},
    {OpponentCharacter::reptile, "reptile", "Reptile"},
    {OpponentCharacter::scorpion, "scorpion", "Scorpion"},
    {OpponentCharacter::sektor, "sektor", "Sektor"},
    {OpponentCharacter::shang_tsung, "shang_tsung", "Shang Tsung"},
    {OpponentCharacter::shao_kahn, "shao_kahn", "Shao Kahn"},
    {OpponentCharacter::sheeva, "sheeva", "Sheeva"},
    {OpponentCharacter::sindel, "sindel", "Sindel"},
    {OpponentCharacter::smoke, "smoke", "Smoke"},
    {OpponentCharacter::human_smoke, "human_smoke", "Human Smoke"},
    {OpponentCharacter::sonya, "sonya", "Sonya Blade"},
    {OpponentCharacter::stryker, "stryker", "Stryker"},
    {OpponentCharacter::subzero, "subzero", "Sub-Zero"},
    {
        OpponentCharacter::classic_subzero,
        "classic_subzero",
        "Classic Sub-Zero"
    },
}};

static_assert(
    kOpponentCharacters.size() == kOpponentCharacterFeatureCount
);

bool program_uses_direction(
    const std::vector<std::vector<int>>& states,
    int forward,
    int back
) {
    return std::any_of(states.begin(), states.end(), [&](const auto& state) {
        return std::find(state.begin(), state.end(), forward) != state.end() ||
            std::find(state.begin(), state.end(), back) != state.end();
    });
}

void mirror_program(
    std::vector<std::vector<int>>& states,
    int forward,
    int back
) {
    for (auto& state : states) {
        for (int& command : state) {
            if (command == forward) {
                command = back;
            } else if (command == back) {
                command = forward;
            }
        }
    }
}

bool same_controller_program(
    const MockMoveDefinition& first,
    const MockMoveDefinition& second
) {
    return first.states == second.states;
}

void append_spatial_mirrors(
    std::vector<MockMoveDefinition>& moves,
    int forward,
    int back
) {
    const std::size_t original_count = moves.size();
    std::vector<MockMoveDefinition> mirrors;
    for (std::size_t index = 0; index < original_count; ++index) {
        const bool directional =
            program_uses_direction(moves[index].states, forward, back);
        if (!directional) {
            continue;
        }

        MockMoveDefinition mirror = moves[index];
        mirror_program(mirror.states, forward, back);
        const bool already_known =
            std::any_of(
                moves.begin(), moves.end(), [&](const auto& known) {
                    return same_controller_program(known, mirror);
                }
            ) ||
            std::any_of(
                mirrors.begin(), mirrors.end(), [&](const auto& known) {
                    return same_controller_program(known, mirror);
                }
            );
        if (already_known) {
            continue;
        }

        mirror.name += " [Espelho F/B]";
        const float family_weight = moves[index].curriculum_weight;
        mirror.curriculum_weight = family_weight * 0.5f;
        moves[index].curriculum_weight =
            family_weight - mirror.curriculum_weight;
        mirrors.push_back(std::move(mirror));
    }
    moves.insert(
        moves.end(),
        std::make_move_iterator(mirrors.begin()),
        std::make_move_iterator(mirrors.end())
    );
}

MockMoveDefinition make_move(
    std::string name,
    std::vector<std::vector<int>> states,
    std::string phase = "special"
) {
    if (states.empty()) {
        throw std::invalid_argument("mock move must contain controller states");
    }
    std::vector<std::string> phases;
    phases.reserve(states.size());
    for (std::size_t index = 0; index < states.size(); ++index) {
        if (states[index].empty()) {
            phases.push_back(
                index + 1U == states.size() ? "observe" : "release"
            );
        } else {
            phases.push_back(phase);
        }
    }
    return MockMoveDefinition{
        .name = std::move(name),
        .states = std::move(states),
        .phases = std::move(phases),
    };
}

std::vector<MockMoveDefinition> with_shared_combat_moves(
    std::vector<MockMoveDefinition> moves,
    FighterFacing facing
) {
    const int forward =
        facing == FighterFacing::right ? kRight : kLeft;
    moves.push_back(make_move(
        "Uppercut agachado",
        {
            {kDown},
            {kDown},
            {kDown, kYHighPunch},
            {kDown, kYHighPunch},
            {},
            {},
        },
        "uppercut"
    ));
    moves.push_back(make_move(
        "Salto para frente com chute alto",
        {
            {kUp, forward},
            {kUp, forward},
            {forward},
            {forward, kXHighKick},
            {kXHighKick},
            {},
            {},
        },
        "air_attack"
    ));
    return moves;
}

std::string normalized_character_slug(std::string_view value) {
    std::string normalized;
    normalized.reserve(value.size());
    bool separator_pending = false;
    for (const unsigned char character : value) {
        if (std::isalnum(character)) {
            if (separator_pending && !normalized.empty()) {
                normalized.push_back('_');
            }
            normalized.push_back(
                static_cast<char>(std::tolower(character))
            );
            separator_pending = false;
        } else {
            separator_pending = !normalized.empty();
        }
    }
    return normalized;
}

std::vector<MockMoveDefinition> make_noob_saibot_moves(
    FighterFacing facing
) {
    const int forward = facing == FighterFacing::right ? kRight : kLeft;
    return {
        make_move(
            "Arremesso de Sombra",
            {{forward}, {}, {forward}, {kYHighPunch}, {}, {}}
        ),
        make_move(
            "Teleporte com Arremesso",
            {{kDown}, {kUp}, {}, {}}
        ),
        make_move(
            "Nuvem de Desativação",
            {{kDown}, {forward}, {kBLowPunch}, {}, {}}
        ),
        make_move(
            "4 Golpes (Y, Y, B, A)",
            {
                {kYHighPunch}, {}, {kYHighPunch}, {},
                {kBLowPunch}, {}, {kALowKick}, {}, {}
            },
            "combo"
        ),
        make_move(
            "4 Golpes (X, A, A, X)",
            {
                {kXHighKick}, {}, {kALowKick}, {},
                {kALowKick}, {}, {kXHighKick}, {}, {}
            },
            "combo"
        ),
    };
}

std::vector<MockMoveDefinition> make_scorpion_moves(
    FighterFacing facing
) {
    const int back = facing == FighterFacing::right ? kLeft : kRight;
    return {
        make_move(
            "Arpão",
            {{back}, {}, {back}, {kBLowPunch}, {}, {}}
        ),
        make_move(
            "Soco Teleportado",
            {{kDown}, {back}, {kYHighPunch}, {}, {}}
        ),
        make_move(
            "Soco Teleportado (Aéreo)",
            {
                {kUp}, {kUp}, {}, {kDown}, {back},
                {kYHighPunch}, {}, {}
            }
        ),
        make_move(
            "Agarrão Aéreo",
            {{kUp}, {kUp}, {}, {kBlock}, {kBlock}, {}, {}}
        ),
        make_move(
            "4 Golpes (Y, Y, X, B+X)",
            {
                {kYHighPunch}, {}, {kYHighPunch}, {},
                {kXHighKick}, {}, {back, kXHighKick}, {}, {}
            },
            "combo"
        ),
        make_move(
            "3 Golpes (X, X, A, B+X)",
            {
                {kXHighKick}, {}, {kXHighKick}, {},
                {kALowKick}, {}, {back, kXHighKick}, {}, {}
            },
            "combo"
        ),
    };
}

std::vector<MockMoveDefinition> make_sindel_moves(
    FighterFacing facing
) {
    const int forward = facing == FighterFacing::right ? kRight : kLeft;
    const int back = facing == FighterFacing::right ? kLeft : kRight;
    return {
        make_move(
            "Bola de Fogo",
            {{forward}, {}, {forward}, {kBLowPunch}, {}, {}}
        ),
        make_move(
            "Bola de Fogo (Aérea)",
            {
                {kUp}, {kUp}, {}, {forward}, {}, {forward},
                {kBLowPunch}, {}, {}
            }
        ),
        make_move(
            "Grito",
            {
                {forward}, {}, {forward}, {}, {forward},
                {kYHighPunch}, {}, {}
            }
        ),
        make_move(
            "Levitar",
            {{back}, {}, {back}, {forward}, {kXHighKick}, {}, {}}
        ),
        make_move(
            "3 Golpes (Y, Y, D+Y)",
            {
                {kYHighPunch}, {}, {kYHighPunch}, {},
                {kDown, kYHighPunch}, {}, {}
            },
            "combo"
        ),
        make_move(
            "4 Golpes (X, Y, Y, B)",
            {
                {kXHighKick}, {}, {kYHighPunch}, {},
                {kYHighPunch}, {}, {kBLowPunch}, {}, {}
            },
            "combo"
        ),
        make_move(
            "Combo Alternativo (X, X, B+X)",
            {
                {kXHighKick}, {}, {kXHighKick}, {},
                {back, kXHighKick}, {}, {}
            },
            "combo"
        ),
    };
}

}  // namespace

std::optional<FighterCharacter> parse_fighter_character(
    std::string_view value
) {
    const std::string normalized = normalized_character_slug(value);
    if (normalized == "subzero" || normalized == "sub_zero") {
        return FighterCharacter::subzero;
    }
    if (normalized == "scorpion") {
        return FighterCharacter::scorpion;
    }
    if (normalized == "noob_saibot" || normalized == "noobsaibot") {
        return FighterCharacter::noob_saibot;
    }
    if (normalized == "sindel") {
        return FighterCharacter::sindel;
    }
    return std::nullopt;
}

std::string_view fighter_character_slug(FighterCharacter character) {
    switch (character) {
        case FighterCharacter::subzero: return "subzero";
        case FighterCharacter::scorpion: return "scorpion";
        case FighterCharacter::noob_saibot: return "noob_saibot";
        case FighterCharacter::sindel: return "sindel";
    }
    throw std::invalid_argument("unsupported fighter character");
}

std::string_view fighter_character_name(FighterCharacter character) {
    switch (character) {
        case FighterCharacter::subzero: return "Sub-Zero";
        case FighterCharacter::scorpion: return "Scorpion";
        case FighterCharacter::noob_saibot: return "Noob Saibot";
        case FighterCharacter::sindel: return "Sindel";
    }
    throw std::invalid_argument("unsupported fighter character");
}

std::vector<std::string> supported_fighter_characters() {
    return {"subzero", "scorpion", "noob_saibot", "sindel"};
}

std::optional<OpponentCharacter> parse_opponent_character(
    std::string_view value
) {
    std::string normalized = normalized_character_slug(value);
    if (normalized == "sub_zero") {
        normalized = "subzero";
    } else if (
        normalized == "classic_sub_zero" ||
        normalized == "subzero_classic" ||
        normalized == "sub_zero_classic"
    ) {
        normalized = "classic_subzero";
    } else if (normalized == "noobsaibot") {
        normalized = "noob_saibot";
    } else if (normalized == "humansmoke" || normalized == "smoke_human") {
        normalized = "human_smoke";
    } else if (normalized == "sonyablade" || normalized == "sonya_blade") {
        normalized = "sonya";
    } else if (normalized == "kunglao") {
        normalized = "kung_lao";
    } else if (normalized == "liukang") {
        normalized = "liu_kang";
    } else if (normalized == "shangtsung") {
        normalized = "shang_tsung";
    } else if (normalized == "shaokahn") {
        normalized = "shao_kahn";
    } else if (normalized == "montaro") {
        normalized = "motaro";
    }

    const auto found = std::find_if(
        kOpponentCharacters.begin(),
        kOpponentCharacters.end(),
        [&](const OpponentCharacterRecord& record) {
            return record.slug == normalized;
        }
    );
    return found == kOpponentCharacters.end()
        ? std::nullopt
        : std::optional<OpponentCharacter>(found->character);
}

std::string_view opponent_character_slug(OpponentCharacter character) {
    const std::size_t index = static_cast<std::size_t>(character);
    if (index >= kOpponentCharacters.size()) {
        throw std::invalid_argument("unsupported opponent character");
    }
    return kOpponentCharacters[index].slug;
}

std::string_view opponent_character_name(OpponentCharacter character) {
    const std::size_t index = static_cast<std::size_t>(character);
    if (index >= kOpponentCharacters.size()) {
        throw std::invalid_argument("unsupported opponent character");
    }
    return kOpponentCharacters[index].name;
}

std::vector<std::string> supported_opponent_characters() {
    std::vector<std::string> characters;
    characters.reserve(kOpponentCharacters.size());
    for (const auto& record : kOpponentCharacters) {
        characters.emplace_back(record.slug);
    }
    return characters;
}

std::vector<MockMoveDefinition> make_sub_zero_mock_moves(FighterFacing facing) {
    const int forward = facing == FighterFacing::right ? kRight : kLeft;
    const int back = facing == FighterFacing::right ? kLeft : kRight;

    return with_shared_combat_moves({
        {
            .name = "Congelar",
            .states = {{kDown}, {forward}, {kBLowPunch}, {}, {}},
            .phases = {"special", "special", "special", "release", "observe"},
        },
        {
            .name = "Clone de Gelo",
            .states = {{kDown}, {back}, {kBLowPunch}, {}, {}},
            .phases = {"special", "special", "special", "release", "observe"},
        },
        {
            .name = "Chuva de Gelo (Frente)",
            .states = {
                {kDown}, {forward}, {kBLowPunch}, {kYHighPunch}, {}, {}
            },
            .phases = {
                "special", "special", "special", "special", "release", "observe"
            },
        },
        {
            .name = "Chuva de Gelo (Perto)",
            .states = {
                {kDown}, {back}, {forward}, {kYHighPunch}, {}, {}
            },
            .phases = {
                "special", "special", "special", "special", "release", "observe"
            },
        },
        {
            .name = "3 Golpes",
            .states = {
                {kYHighPunch},
                {},
                {kYHighPunch},
                {},
                {kDown, kBLowPunch},
                {},
                {},
            },
            .phases = {
                "combo", "release", "combo", "release",
                "combo", "release", "observe"
            },
        },
        {
            .name = "4 Golpes",
            .states = {
                {kXHighKick},
                {},
                {kXHighKick},
                {},
                {back, kXHighKick},
                {},
                {forward, kALowKick},
                {},
                {},
            },
            .phases = {
                "combo", "release", "combo", "release", "combo",
                "release", "combo", "release", "observe"
            },
        },
    }, facing);
}

std::vector<MockMoveDefinition> make_character_mock_moves(
    FighterCharacter character,
    FighterFacing facing
) {
    switch (character) {
        case FighterCharacter::subzero:
            return make_sub_zero_mock_moves(facing);
        case FighterCharacter::scorpion:
            return with_shared_combat_moves(
                make_scorpion_moves(facing), facing
            );
        case FighterCharacter::noob_saibot:
            return with_shared_combat_moves(
                make_noob_saibot_moves(facing), facing
            );
        case FighterCharacter::sindel:
            return with_shared_combat_moves(
                make_sindel_moves(facing), facing
            );
    }
    throw std::invalid_argument("unsupported fighter character");
}

std::vector<MockMoveDefinition> make_sub_zero_curriculum(
    FighterFacing facing
) {
    const int forward = facing == FighterFacing::right ? kRight : kLeft;
    const int back = facing == FighterFacing::right ? kLeft : kRight;
    auto moves = make_sub_zero_mock_moves(facing);
    const float family_weight =
        100.0f / static_cast<float>(moves.size());
    for (auto& move : moves) {
        move.curriculum_weight = family_weight;
    }
    append_spatial_mirrors(moves, forward, back);
    return moves;
}

std::vector<MockMoveDefinition> make_character_curriculum(
    FighterCharacter character,
    FighterFacing facing
) {
    const int forward = facing == FighterFacing::right ? kRight : kLeft;
    const int back = facing == FighterFacing::right ? kLeft : kRight;
    auto moves = make_character_mock_moves(character, facing);
    const float family_weight =
        100.0f / static_cast<float>(moves.size());
    for (auto& move : moves) {
        move.curriculum_weight = family_weight;
    }
    append_spatial_mirrors(moves, forward, back);
    return moves;
}

MockAgent::MockAgent(
    std::optional<std::uint32_t> random_seed,
    FighterFacing facing,
    FighterCharacter character
)
    : moves_(make_character_curriculum(character, facing)),
      random_engine_(random_seed.value_or(std::random_device{}())) {}

void MockAgent::select_next_move() {
    std::vector<double> weights;
    weights.reserve(moves_.size());
    for (const auto& move : moves_) {
        weights.push_back(
            disable_cloud_cooldown_moves_remaining_ > 0 &&
                    is_disable_cloud_move(move.name)
                ? 0.0
                : std::max(0.0f, move.curriculum_weight)
        );
    }
    std::discrete_distribution<std::size_t> move_distribution(
        weights.begin(), weights.end()
    );
    active_move_ = move_distribution(random_engine_);
    const auto& move = moves_[*active_move_];
    if (disable_cloud_cooldown_moves_remaining_ > 0) {
        --disable_cloud_cooldown_moves_remaining_;
    }
    if (is_disable_cloud_move(move.name)) {
        disable_cloud_cooldown_moves_remaining_ =
            kDisableCloudCooldownMoves;
    }
    active_states_ = move.states;
    active_phases_ = move.phases;
    active_step_ = 0;
}

MockAgentDecision MockAgent::infer(
    const std::vector<std::shared_ptr<const BufferedFrame>>& observation,
    const std::vector<AppliedActionRecord>& applied_action_history
) {
    if (observation.empty()) {
        throw std::invalid_argument("mock inference requires at least one frame");
    }
    static_cast<void>(applied_action_history);
    std::lock_guard lock(mutex_);

    if (!active_move_.has_value()) {
        select_next_move();
    }

    const MockMoveDefinition& move = moves_[*active_move_];
    MockAgentDecision decision{
        .commands = active_states_[active_step_],
        .move_name = move.name,
        .step_index = active_step_ + 1U,
        .step_count = active_states_.size(),
        .phase =
            active_step_ < active_phases_.size()
                ? active_phases_[active_step_]
                : "unspecified",
        .action_source = move.action_source,
    };

    ++active_step_;
    if (active_step_ == active_states_.size()) {
        active_move_.reset();
        active_states_.clear();
        active_phases_.clear();
        active_step_ = 0;
    }
    return decision;
}

void MockAgent::reset_sequence() {
    std::lock_guard lock(mutex_);
    active_move_.reset();
    active_states_.clear();
    active_phases_.clear();
    active_step_ = 0;
    disable_cloud_cooldown_moves_remaining_ = 0;
}

}  // namespace gameagent
