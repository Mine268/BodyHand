#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <opencv2/opencv.hpp>

#include "argparse.h"
#include "calibCheckboard.h"

std::string makeOutputPath(const std::string path) {
	std::filesystem::path p(path);
	auto dir = p.parent_path();
	auto outFile = dir / "output.txt";
	return outFile.string();
}

int NUM_HEIGHT = 0;
int NUM_WIDTH = 0;
float SQUARE_SIZE = 0.0;

cv::Mat intr = (cv::Mat_<double>(3, 3) <<
	1045.977, 0, 693.407,
	0, 1042.865, 581.199,
	0, 0, 1);
cv::Mat dist = (cv::Mat_<double>(1, 5) << -0.0751, -0.1446, 0, 0, 0.2794);

int main(int argc, char** argv) {
	argparse::ArgumentParser parser("Calibration");
	parser.add_argument("checkboard_image").help("image path");
	parser.add_argument("checkboard_num_height").help("Inner").scan<'i', int>();
	parser.add_argument("checkboard_num_width").help("Inner").scan<'i', int>();
	parser.add_argument("checkboard_size").help("in mm").scan<'g', float>();

	std::string checkboard_path;

	try {
		parser.parse_args(argc, argv);
		checkboard_path = parser.get<std::string>("checkboard_image");
		NUM_HEIGHT = parser.get<int>("checkboard_num_height");
		NUM_WIDTH = parser.get<int>("checkboard_num_width");
		SQUARE_SIZE = parser.get<float>("checkboard_size");
	}
	catch (const std::runtime_error& err) {
		std::cerr << err.what() << std::endl;
	}

	cv::Mat checkboard_img = cv::imread(checkboard_path, cv::IMREAD_COLOR);
	if (checkboard_img.empty()) {
		std::cerr << "Failed to load image: " << checkboard_img << std::endl;
		return -1;
	}

	cv::Vec3d rvec, tvec;
	cv::Mat image_out;
	double reproj_err = 0.0f;
	bool result = estimatePoseFromChessboard(
		checkboard_img,
		cv::Size(NUM_HEIGHT, NUM_WIDTH),
		SQUARE_SIZE,
		intr,
		dist,
		rvec,
		tvec,
		true,
		&image_out,
		nullptr);
	if (!result) {
		std::cerr << "Calibration failed" << std::endl;
		return -1;
	}

	cv::Mat pmat = poseRtTo44(rvec, tvec);
	std::cout << pmat << std::endl;

	std::ofstream outFile(makeOutputPath(checkboard_path));
	if (outFile.is_open()) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				outFile << " " + (i == 0 && j == 0) << pmat.at<double>(i, j);
			}
		}
		for (int i = 0; i < 3; ++i) {
			outFile << " " << pmat.at<double>(i, 3);
		}
		std::cout << std::endl;
		std::cout << "result written to: " << makeOutputPath(checkboard_path) << std::endl;
		outFile.close();
	}
	else {
		std::cerr << "Failed to open output file: " << makeOutputPath(checkboard_path) << std::endl;
	}

	std::cout << "reproj error: " << reproj_err << std::endl;
	cv::imshow("reproj", image_out);
	cv::waitKey(0);

	return 0;
}