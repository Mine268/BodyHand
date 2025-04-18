#include <filesystem>
#include <iostream>
#include "multiCameraCalibration.h"
#include "visualization.h"

int main() {
	std::string calibrationRoot = "E:\\hand_stereo_cap\\cpp_calib_stereo2";
	std::string viewName = "V";
	bool result = multiCameraCalibrationAndSave(calibrationRoot, viewName, 54, true);
	if (!result) {
		std::cerr << "Calibration failed" << std::endl;
	}
	return 0;
}