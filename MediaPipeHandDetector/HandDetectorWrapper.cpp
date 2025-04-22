#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "HandDetectorWrapper.h"

namespace py = pybind11;

struct HandDetectorWrapper::Impl {
    py::object detector;
};

HandDetectorWrapper::HandDetectorWrapper(const std::string& model_path) {
    // Initialize Python interpreter if not already done
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }

    py::module_ hand_detector_module = py::module_::import("hand_detector_module");
    impl_ = std::make_unique<Impl>();
    impl_->detector = hand_detector_module.attr("HandDetector")(model_path);
}

HandDetectorWrapper::~HandDetectorWrapper() {
    // No need to explicitly close Python interpreter here
    // It will be handled by pybind11
}

HandDetectorWrapper::HandResult HandDetectorWrapper::estimate(
    const std::vector<uint8_t>& image_data,
    int width, int height, int64_t timestamp_ms) {

    // Convert image data to numpy array
    py::array_t<uint8_t> img_array({ height, width, 3 }, image_data.data());

    // Call Python method
    py::object result = impl_->detector.attr("estimate")(img_array, timestamp_ms);

    // Convert results to C++ types
    HandResult cpp_result;

    auto convert_hand = [](py::object py_hand) -> std::vector<std::vector<float>> {
        if (py_hand.is_none()) {
            return {};
        }

        py::array_t<float> np_hand = py_hand.cast<py::array_t<float>>();
        auto buf = np_hand.request();
        float* ptr = static_cast<float*>(buf.ptr);

        size_t num_points = buf.shape[0];
        size_t dims = buf.shape[1];

        std::vector<std::vector<float>> hand_data(num_points, std::vector<float>(dims));

        for (size_t i = 0; i < num_points; ++i) {
            for (size_t j = 0; j < dims; ++j) {
                hand_data[i][j] = ptr[i * dims + j];
            }
        }

        return hand_data;
        };

    cpp_result.leftHand = convert_hand(result.attr("leftHand"));
    cpp_result.rightHand = convert_hand(result.attr("rightHand"));
    cpp_result.leftHand25D = convert_hand(result.attr("leftHand_25D"));
    cpp_result.rightHand25D = convert_hand(result.attr("rightHand_25D"));

    return cpp_result;
}