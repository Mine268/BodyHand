#pragma once
#include <string>

/**
 * ����ͼ����궨
 */

bool multiCameraCalibrationAndSave(std::string& calibrationDir,
	std::string& viewName, int cornersCountPerImage, bool is_fix_instrinic);
