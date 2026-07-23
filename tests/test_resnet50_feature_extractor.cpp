#include "ResNet50FeatureExtractor.h"

#include <image/image_decode.h>

#include <json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

std::string environment_or(const char* name, std::string fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? value : std::move(fallback);
}

std::size_t feature_limit() {
    const std::string value = environment_or("GAMEAGENT_FEATURE_LIMIT", "0");
    return static_cast<std::size_t>(std::stoull(value));
}

struct CaptureRecord {
    std::filesystem::path image_path;
    std::filesystem::path metadata_path;
    std::uint64_t frame_id = 0;
    std::int64_t client_timestamp_ms = 0;
    std::int64_t received_timestamp_ms = 0;
};

std::vector<CaptureRecord> find_capture_images(
    const std::filesystem::path& root,
    std::size_t limit
) {
    std::vector<CaptureRecord> result;
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file() ||
            entry.path().filename() != "frames.jsonl") {
            continue;
        }
        std::ifstream metadata(entry.path());
        if (!metadata) {
            throw std::runtime_error(
                "could not read capture metadata: " + entry.path().string()
            );
        }
        std::string line;
        while (std::getline(metadata, line)) {
            if (line.empty()) {
                continue;
            }
            const nlohmann::json record = nlohmann::json::parse(line);
            if (!record.contains("file") ||
                !record["file"].is_string() ||
                !record.contains("frame_id") ||
                !record["frame_id"].is_number_unsigned() ||
                !record.contains("client_timestamp_ms") ||
                !record["client_timestamp_ms"].is_number_integer() ||
                !record.contains("received_timestamp_ms") ||
                !record["received_timestamp_ms"].is_number_integer()) {
                throw std::runtime_error(
                    "invalid capture metadata record: " + entry.path().string()
                );
            }
            const std::filesystem::path image_path =
                entry.path().parent_path() /
                record["file"].get<std::string>();
            if (!std::filesystem::is_regular_file(image_path)) {
                throw std::runtime_error(
                    "capture metadata references a missing image: " +
                    image_path.string()
                );
            }
            result.push_back({
                .image_path = image_path,
                .metadata_path = entry.path(),
                .frame_id = record["frame_id"].get<std::uint64_t>(),
                .client_timestamp_ms =
                    record["client_timestamp_ms"].get<std::int64_t>(),
                .received_timestamp_ms =
                    record["received_timestamp_ms"].get<std::int64_t>(),
            });
        }
    }
    std::sort(
        result.begin(), result.end(),
        [](const CaptureRecord& left, const CaptureRecord& right) {
            if (left.client_timestamp_ms != right.client_timestamp_ms) {
                return left.client_timestamp_ms < right.client_timestamp_ms;
            }
            if (left.metadata_path != right.metadata_path) {
                return left.metadata_path < right.metadata_path;
            }
            return left.frame_id < right.frame_id;
        }
    );
    if (limit != 0 && result.size() > limit) {
        // Keep the newest debug captures when an explicit test limit is used.
        result.erase(result.begin(), result.end() - static_cast<std::ptrdiff_t>(limit));
    }
    return result;
}

struct FeatureStatistics {
    float minimum = 0.0f;
    float maximum = 0.0f;
    double mean = 0.0;
    double l2_norm = 0.0;
};

FeatureStatistics feature_statistics(const std::vector<float>& feature) {
    if (feature.empty()) {
        throw std::runtime_error("empty ResNet50 feature");
    }
    FeatureStatistics statistics{
        .minimum = std::numeric_limits<float>::infinity(),
        .maximum = -std::numeric_limits<float>::infinity(),
    };
    double sum = 0.0;
    double squared_sum = 0.0;
    for (float value : feature) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite ResNet50 feature value");
        }
        statistics.minimum = std::min(statistics.minimum, value);
        statistics.maximum = std::max(statistics.maximum, value);
        sum += value;
        squared_sum += static_cast<double>(value) * value;
    }
    statistics.mean = sum / static_cast<double>(feature.size());
    statistics.l2_norm = std::sqrt(squared_sum);
    return statistics;
}

void write_feature(
    const std::filesystem::path& path,
    const std::vector<float>& feature
) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not create feature file: " + path.string());
    }
    output.write(
        reinterpret_cast<const char*>(feature.data()),
        static_cast<std::streamsize>(feature.size() * sizeof(float))
    );
    if (!output) {
        throw std::runtime_error("could not write feature file: " + path.string());
    }
}

}  // namespace

int main() {
    try {
        const std::filesystem::path checkpoint = environment_or(
            "GAMEAGENT_RESNET50_CHECKPOINT",
            "/workspace/resnet50_best.ckpt"
        );
        const std::filesystem::path capture_root = environment_or(
            "GAMEAGENT_CAPTURE_ROOT",
            "/workspace/gameagent/captures"
        );
        const std::filesystem::path output_root = environment_or(
            "GAMEAGENT_FEATURE_OUTPUT_DIR",
            "/workspace/gameagent/resnet50_features"
        );
        const int device_id = std::stoi(
            environment_or("GAMEAGENT_DEVICE_ID", "0")
        );

        if (!std::filesystem::is_regular_file(checkpoint)) {
            std::cout << "[SKIP] ResNet50 checkpoint not found: "
                      << checkpoint << '\n';
            return 77;
        }
        if (!std::filesystem::is_directory(capture_root)) {
            std::cout << "[SKIP] GameAgent capture directory not found: "
                      << capture_root << '\n';
            return 77;
        }

        const auto images = find_capture_images(capture_root, feature_limit());
        if (images.empty()) {
            std::cout << "[SKIP] no saved debug JPEGs under: "
                      << capture_root << '\n';
            return 77;
        }

        const auto run_id =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
        const std::filesystem::path output_directory =
            output_root /
            ("run_" + std::to_string(run_id) + "_p" + std::to_string(getpid()));
        std::filesystem::create_directories(output_directory);

        gameagent::ResNet50FeatureExtractor extractor(device_id, true);
        const auto report = extractor.load_checkpoint(checkpoint.string());
        if (!extractor.loaded() || report.tensor_count != 320 ||
            report.verification_max_abs != 0.0f) {
            throw std::runtime_error(
                "ResNet50 checkpoint did not pass exact device verification"
            );
        }

        std::ofstream manifest(
            output_directory / "features.jsonl",
            std::ios::trunc
        );
        if (!manifest) {
            throw std::runtime_error("could not create feature manifest");
        }

        double total_inference_ms = 0.0;
        std::vector<float> first_feature;
        for (std::size_t index = 0; index < images.size(); ++index) {
            const auto image =
                base::data::image::decode_image_rgb8(
                    images[index].image_path.string()
                );
            const auto start = std::chrono::steady_clock::now();
            const auto feature = extractor.extract(image);
            const auto end = std::chrono::steady_clock::now();
            const double inference_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
            total_inference_ms += inference_ms;

            if (feature.size() !=
                static_cast<std::size_t>(
                    gameagent::ResNet50FeatureExtractor::kFeatureSize
                )) {
                throw std::runtime_error(
                    "ResNet50 did not return 2048 feature values"
                );
            }
            const FeatureStatistics statistics = feature_statistics(feature);
            if (!(statistics.l2_norm > 0.0)) {
                throw std::runtime_error("ResNet50 returned an all-zero feature");
            }

            std::ostringstream filename;
            filename << "feature_" << std::setw(6) << std::setfill('0')
                     << index << ".f32";
            const std::filesystem::path feature_path =
                output_directory / filename.str();
            write_feature(feature_path, feature);

            manifest << nlohmann::json{
                {"image_path", images[index].image_path.string()},
                {"capture_metadata_path",
                 images[index].metadata_path.string()},
                {"sequence_id",
                 std::filesystem::relative(
                     images[index].metadata_path.parent_path(),
                     capture_root
                 ).string()},
                {"frame_id", images[index].frame_id},
                {"client_timestamp_ms",
                 images[index].client_timestamp_ms},
                {"received_timestamp_ms",
                 images[index].received_timestamp_ms},
                {"feature_path", feature_path.string()},
                {"source_width", image.width()},
                {"source_height", image.height()},
                {"preprocessed_width",
                 gameagent::ResNet50FeatureExtractor::kInputWidth},
                {"preprocessed_height",
                 gameagent::ResNet50FeatureExtractor::kInputHeight},
                {"feature_size",
                 gameagent::ResNet50FeatureExtractor::kFeatureSize},
                {"dtype", "float32"},
                {"inference_ms", inference_ms},
                {"minimum", statistics.minimum},
                {"maximum", statistics.maximum},
                {"mean", statistics.mean},
                {"l2_norm", statistics.l2_norm},
            }.dump() << '\n';

            if (index == 0) {
                first_feature = feature;
            }
            std::cout << "[FEATURE] " << (index + 1U) << '/' << images.size()
                      << " image=" << images[index].image_path
                      << " frame_id=" << images[index].frame_id
                      << " source=" << image.width() << 'x' << image.height()
                      << " dim=" << feature.size()
                      << " l2=" << statistics.l2_norm
                      << " inference_ms=" << inference_ms << '\n';
        }

        const auto repeated_image =
            base::data::image::decode_image_rgb8(
                images.front().image_path.string()
            );
        const auto repeated_feature = extractor.extract(repeated_image);
        float repeat_max_abs = 0.0f;
        for (std::size_t index = 0; index < first_feature.size(); ++index) {
            repeat_max_abs = std::max(
                repeat_max_abs,
                std::abs(first_feature[index] - repeated_feature[index])
            );
        }
        if (repeat_max_abs > 1.0e-5f) {
            throw std::runtime_error(
                "repeated ResNet50 inference is not deterministic"
            );
        }

        const nlohmann::json summary{
            {"checkpoint", checkpoint.string()},
            {"checkpoint_tensors", report.tensor_count},
            {"checkpoint_values", report.value_count},
            {"checkpoint_epoch", report.epoch},
            {"checkpoint_global_step", report.global_step},
            {"checkpoint_best_accuracy", report.best_accuracy},
            {"checkpoint_checksum", report.checksum},
            {"checkpoint_verification_max_abs",
             report.verification_max_abs},
            {"images_processed", images.size()},
            {"feature_size",
             gameagent::ResNet50FeatureExtractor::kFeatureSize},
            {"feature_dtype", "float32"},
            {"preprocessing",
             "bilinear_224x224_chw_imagenet_mean_std"},
            {"average_inference_ms",
             total_inference_ms / static_cast<double>(images.size())},
            {"repeat_inference_max_abs", repeat_max_abs},
            {"device_id", device_id},
        };
        std::ofstream summary_output(
            output_directory / "summary.json",
            std::ios::trunc
        );
        summary_output << std::setw(2) << summary << '\n';
        if (!summary_output) {
            throw std::runtime_error("could not write feature summary");
        }

        std::cout << "[CHECKPOINT] tensors=" << report.tensor_count
                  << " values=" << report.value_count
                  << " epoch=" << report.epoch
                  << " best_accuracy=" << report.best_accuracy
                  << " checksum=" << report.checksum
                  << " verification_max_abs="
                  << report.verification_max_abs << '\n'
                  << "[OUTPUT] " << output_directory << '\n'
                  << "[PASS] ResNet50 checkpoint inference and saved-image "
                     "feature extraction passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
