#define _CRT_SECURE_NO_WARNINGS

#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "argparse.h"
#include "Pose3D.h"
#include "CMultiCap.h"

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



// 可视化窗口
cv::viz::Viz3d window("Pose");

void updateViz(
	const std::vector<cv::Point3f>& body_kps,
	const std::vector<cv::Point3f>& hand_kps,
	const std::vector<std::pair<int, int>>& body_connection = BODY_CONNECTION,
	const std::vector<std::pair<int, int>>& hand_connection = HAND_CONNECTION

) {
	window.removeAllWidgets();
	//window.setBackgroundColor(cv::viz::Color::white());

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

	//window.spin();
	window.spinOnce(1, true);
}

std::string get_time_string() {
	auto now = std::chrono::system_clock::now();
	std::time_t now_c = std::chrono::system_clock::to_time_t(now);
	std::tm* parts = std::localtime(&now_c);
	char buffer[20];
	std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", parts);
	return std::string(buffer);
}

int main(int argc, char** argv) {

	// ******** 参数解析 ********
	std::string config_path;
	argparse::ArgumentParser parser("Pose estimation");
	parser.add_description(
		"从配置文件中读取模型信息和相机标定信息，进行姿态估计。"
		"\n\t第一、二、三行：分别是人体姿态估计模型，人体检测模型，手部姿态估计模型的地址"
		"\n\t第四行是一个正整数n，表示总共有多少个视图"
		"\n\t接下来的n行每行有26个浮点数，前9个表示内参矩阵，接着的9个表示旋转变换矩阵，然后3个表示位移变换向量，最后的5个表示畸变参数"
	);
	parser.add_argument("config_path").help("配置文件地址");
	try {
		parser.parse_args(argc, argv);
		config_path = parser.get<std::string>("config_path");
	}
	catch (const std::runtime_error& err) {
		std::cerr << err.what() << std::endl;
	}

	// ******** 从文件读取初始化配置 ********
	BodyHand::BodyModelConfig body_cfg;
	BodyHand::HandModelConfig hand_cfg;
	int num_view;
	std::vector<cv::Mat> intr;
	std::vector<cv::Mat> rot_trans;
	std::vector<cv::Mat> transl_trans;
	std::vector<std::vector<float>> undist;

	std::ifstream config_file(config_path);
	// 配置文件读取
	if (!config_file.is_open()) {
		std::cerr << "无法打开配置文件：" << config_path << std::endl;
		config_file.close();
		return -1;
	}
	// 模型文件地址读取
	if (!std::getline(config_file, body_cfg.model_path)) {
		std::cerr << "无法读取人体姿态估计模型件地址。" << std::endl;
		config_file.close();
		return -1;
	}
	if (!std::getline(config_file, hand_cfg.handlr_path)) {
		std::cerr << "无法读取人体检测模型文件地址。" << std::endl;
		config_file.close();
		return -1;
	}
	if (!std::getline(config_file, hand_cfg.hamer_path)) {
		std::cerr << "无法读取手部姿态估计模型文件地址。" << std::endl;
		config_file.close();
		return -1;
	}
	// 相机信息读取
	if (!(config_file >> num_view)) {
		std::cerr << "无法读取视图数量。" << std::endl;
		config_file.close();
		return -1;
	}
	// 逐行读取数据
	for (int i = 0; i < num_view; ++i) {
		// 临时数组存储一行26个float
		float data[26];
		for (int j = 0; j < 26; ++j) {
			if (!(config_file >> data[j])) { // 读取失败处理
				std::cerr << "文件格式错误或数据不足。" << std::endl;
				config_file.close();
				return -1;
			}
		}
		// 1. 内参矩阵 3x3 (前9个元素)
		intr.emplace_back(cv::Mat(3, 3, CV_32F, data).clone());
		// 2. 旋转矩阵 3x3 (中间9个元素)
		rot_trans.emplace_back(cv::Mat(3, 3, CV_32F, data + 9).clone());
		// 3. 平移向量 3x1 (接下来3个元素)
		transl_trans.emplace_back(cv::Mat(3, 1, CV_32F, data + 18).clone());
		// 4. 畸变参数 5个 (最后5个元素)
		undist.emplace_back(std::vector<float> {data[21], data[22], data[23], data[24], data[25]});
	}
	config_file.close();

	// ******** 构造姿态估计器 ********
	BodyHand::PoseEstimator pe{
		body_cfg,
		hand_cfg,
		num_view,
		intr,
		rot_trans,
		transl_trans,
		undist
	};

	// ******** 构造捕捉系统 ********
	get_app();
	init_device();
	if (num_view != get_device_count()) {
		std::cerr << "配置文件的视图数目为：" << num_view << "，而接入的相机数目为：" << get_device_count() << std::endl;
		close_device();
		return -1;
	}
	start_grabbing();
	
	// ******** 初始化可视化窗口 ********
	window.setBackgroundColor(cv::viz::Color::white());
	window.spinOnce(1, true);

	// ******** 捕捉和姿态估计 ********
	std::ofstream pose_ofs(get_time_string() + "_pose_output.txt"); // 按照当前日期命名的姿态输出文件
	do {
		auto cap_info = capture();
		if (cap_info.flag) {
			std::vector<cv::Mat> imgs;
			BodyHand::PoseResult pose_result;
			for (int i = 0; i < num_view; ++i) {
				cv::Mat img_rgb(cap_info.height[i], cap_info.width[i], CV_8UC3, cap_info.ppbuffer[i]), img_bgr;
				cv::cvtColor(img_rgb, img_bgr, cv::COLOR_RGB2BGR);
				cv::imshow(std::format("{}", i), img_bgr);
				imgs.emplace_back(std::move(img_bgr));
			}
			cv::waitKey(1);

			pe.estimatePose(imgs, pose_result, 0);
			if (pose_result.valid_body && pose_result.valid_left && pose_result.valid_right) {
				std::cout << "Found pose\n";
				std::string time_string = get_time_string();
				// 可视化到opencv
				updateViz(pose_result.body_kps_3d, pose_result.hand_kps_3d);
				// 存储到文件，每行第一个是时间戳
				pose_ofs << time_string << ' ';
				// 然后是全部3d人体关节点坐标
				for (const auto& kp : pose_result.body_kps_3d) {
					pose_ofs << kp.x << ' ' << kp.y << ' ' << kp.z << ' ';
				}
				// 然后是全部42个手部关节点坐标，先左后右
				for (const auto& kp : pose_result.hand_kps_3d) {
					pose_ofs << kp.x << ' ' << kp.y << ' ' << kp.z << ' ';
				}
				pose_ofs << '\n';
			}
			if (window.wasStopped()) {
				break;
			}
			cv::waitKey(50);
		}
	} while (true);
	pose_ofs.close();

	// ******** 停止捕捉和释放资源 ********
	stop_grabbing();
	close_device();

	return 0;
}