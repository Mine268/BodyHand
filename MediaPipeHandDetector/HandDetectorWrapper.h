#pragma once

#include <string>
#include <memory>
#include <vector>

class HandDetectorWrapper {
public:
    HandDetectorWrapper(const std::string& model_path);
    ~HandDetectorWrapper();

    struct HandResult {
        std::vector<std::vector<float>> leftHand;
        std::vector<std::vector<float>> rightHand;
        std::vector<std::vector<float>> leftHand25D;
        std::vector<std::vector<float>> rightHand25D;
    };

    HandResult estimate(
        const std::vector<uint8_t>& image_data,
        int width,
        int height,
        int64_t timestamp_ms
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};