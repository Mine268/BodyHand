#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include "Pose3D.h"

const std::vector<std::pair<int, int>> BODY_CONNECTION{
	{0, 1}, {0, 2}, {1, 3}, {2, 4},
	{5, 6}, {6, 12}, {12, 11}, {11, 5},
	{5, 7}, {7, 9}, {6, 8}, {8, 10},
	{11, 13}, {13, 15}, {12, 14}, {14, 16}
};
const std::vector<std::pair<int, int>> HAND_CONNECTION{
	{0, 1}, {1, 2}, {2, 3}, {3, 4},
	{5, 9}, {9, 13}, {13, 17}, {17, 0}, {0, 5},
	{5, 6}, {6, 7}, {7, 8},
	{9, 10}, {10, 11}, {11, 12},
	{13, 14}, {14, 15}, {15, 16},
	{17, 18}, {18, 19}, {19, 20}
};

void viz(
	const std::vector<cv::Point3f>& body_kps,
	const std::vector<cv::Point3f>& hand_kps,
	const std::vector<std::pair<int, int>>& body_connection = BODY_CONNECTION,
	const std::vector<std::pair<int, int>>& hand_connection = HAND_CONNECTION

) {
	cv::viz::Viz3d window("Pose");
	window.setBackgroundColor(cv::viz::Color::white());

	cv::Mat body_points_mat(body_kps.size(), 1, CV_32FC3);
	for (size_t i = 0; i < body_kps.size(); ++i) {
		body_points_mat.at<cv::Vec3f>(i, 0) = cv::Vec3f(
			body_kps[i].x,
			body_kps[i].y,
			body_kps[i].z
		);
	}

	// 1. 显示所有点
	for (size_t i = 0; i < body_kps.size(); ++i) {
		// 创建小球体：参数为(中心坐标, 半径, 细分精度, 颜色)
		cv::viz::WSphere sphere(
			body_kps[i], // 点坐标
			10.0, // 半径（控制粗细）
			10, // 细分精度（值越高越圆滑）
			cv::viz::Color::red() // 颜色
		);
		window.showWidget("BodySphere_" + std::to_string(i), sphere);
	}
	for (size_t i = 0; i < hand_kps.size(); ++i) {
		cv::viz::WSphere sphere(
			hand_kps[i],
			7.0,
			10,
			(i < 21) ? cv::viz::Color::blue() : cv::viz::Color::green()
		);
		window.showWidget("HandSphere_" + std::to_string(i), sphere);
	}

	// 2. 绘制连接线
	for (const auto& conn : body_connection) {
		int idx1 = conn.first;
		int idx2 = conn.second;

		// 创建线段（起点，终点，颜色）
		cv::viz::WLine line(
			body_kps[idx1],
			body_kps[idx2],
			cv::viz::Color::gray()
		);
		window.showWidget("BodyLine_" + std::to_string(idx1) + "_" + std::to_string(idx2), line);
	}
	for (const auto& conn : hand_connection) {
		int idx1 = conn.first;
		int idx2 = conn.second;

		cv::viz::WLine lineL(
			hand_kps[idx1],
			hand_kps[idx2],
			cv::viz::Color::gray()
		);
		cv::viz::WLine lineR(
			hand_kps[idx1 + 21],
			hand_kps[idx2 + 21],
			cv::viz::Color::gray()
		);
		window.showWidget("HandLLine_" + std::to_string(idx1) + "_" + std::to_string(idx2), lineL);
		window.showWidget("HandRLine_" + std::to_string(idx1) + "_" + std::to_string(idx2), lineR);
	}

	window.spin();
}

int main() {
	BodyHand::BodyModelConfig body_cfg;
	body_cfg.model_path = "../BodyHand/models/yolov8s-pose.onnx";
	BodyHand::HandModelConfig hand_cfg;
	hand_cfg.handlr_path = "../BodyHand/models/handLR_480x640.onnx";
	hand_cfg.hamer_path = "../BodyHand/models/hand_mano.onnx";

	std::vector<cv::Mat> intr = {
		(cv::Mat_<float>(3, 3) << 1880, 0, 720, 0, 1880, 540, 0, 0, 1),
		(cv::Mat_<float>(3, 3) << 1880, 0, 720, 0, 1880, 540, 0, 0, 1)
	};
	std::vector<cv::Mat> rot = {
		(cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1),
		(cv::Mat_<float>(3, 3) <<
			0.8683880299764319, -0.1903773988860543, 0.4578850023608894,
			0.168825953174351, 0.981710571921911, 0.08798949091525754,
			-0.4662617579519291, 0.0008938512932874579, 0.8846463553880963
		)
	};
	std::vector<cv::Mat> transl = {
		(cv::Mat_<float>(3, 1) << 0, 0, 0),
		(cv::Mat_<float>(3, 1) << -1132.757016065035, -147.9197843434138, 715.4395357461926)
	};
	std::vector<std::vector<float>> undist = {
		{-0.0287, -0.9157999999999999, 0, 0, 0},
		{-0.0287, -0.9157999999999999, 0, 0, 0}
	};

	BodyHand::PoseEstimator pe{
		body_cfg,
		hand_cfg,
		2,
		intr,
		rot,
		transl,
		undist
	};

	std::vector<cv::Mat> imgs = {
		cv::imread(R"(E:\BodyHandCapture\2025_04_22_1\frames\V0\000465.jpg)"),
		cv::imread(R"(E:\BodyHandCapture\2025_04_22_1\frames\V1\000465.jpg)")
	};

	BodyHand::PoseResult pose_result;
	pe.estimatePose(imgs, pose_result);
	std::cout << 
		std::format("body: {}, right: {}, left: {}", pose_result.valid_body, pose_result.valid_right, pose_result.valid_left) <<
		std::endl;
	viz(pose_result.body_kps_3d, pose_result.hand_kps_3d);

	return 0;
}