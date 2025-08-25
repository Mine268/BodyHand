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



// ���ӻ�����
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

	// 1. ��ʾ���е�
	for (size_t i = 0; i < body_kps.size(); ++i) {
		// ����С���壺����Ϊ(��������, �뾶, ϸ�־���, ��ɫ)
		cv::viz::WSphere sphere(
			body_kps[i], // ������
			10.0, // �뾶�����ƴ�ϸ��
			10, // ϸ�־��ȣ�ֵԽ��ԽԲ����
			cv::viz::Color::red() // ��ɫ
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

	// 2. ����������
	for (const auto& conn : body_connection) {
		int idx1 = conn.first;
		int idx2 = conn.second;

		// �����߶Σ���㣬�յ㣬��ɫ��
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

	// ******** �������� ********
	std::string config_path;
	argparse::ArgumentParser parser("Pose estimation");
	parser.add_description(
		"�������ļ��ж�ȡģ����Ϣ������궨��Ϣ��������̬���ơ�"
		"\n\t��һ���������У��ֱ���������̬����ģ�ͣ�������ģ�ͣ��ֲ���̬����ģ�͵ĵ�ַ"
		"\n\t��������һ��������n����ʾ�ܹ��ж��ٸ���ͼ"
		"\n\t��������n��ÿ����26����������ǰ9����ʾ�ڲξ��󣬽��ŵ�9����ʾ��ת�任����Ȼ��3����ʾλ�Ʊ任����������5����ʾ�������"
	);
	parser.add_argument("config_path").help("�����ļ���ַ");
	try {
		parser.parse_args(argc, argv);
		config_path = parser.get<std::string>("config_path");
	}
	catch (const std::runtime_error& err) {
		std::cerr << err.what() << std::endl;
	}

	// ******** ���ļ���ȡ��ʼ������ ********
	BodyHand::BodyModelConfig body_cfg;
	BodyHand::HandModelConfig hand_cfg;
	int num_view;
	std::vector<cv::Mat> intr;
	std::vector<cv::Mat> rot_trans;
	std::vector<cv::Mat> transl_trans;
	std::vector<std::vector<float>> undist;

	std::ifstream config_file(config_path);
	// �����ļ���ȡ
	if (!config_file.is_open()) {
		std::cerr << "�޷��������ļ���" << config_path << std::endl;
		config_file.close();
		return -1;
	}
	// ģ���ļ���ַ��ȡ
	if (!std::getline(config_file, body_cfg.model_path)) {
		std::cerr << "�޷���ȡ������̬����ģ�ͼ���ַ��" << std::endl;
		config_file.close();
		return -1;
	}
	if (!std::getline(config_file, hand_cfg.handlr_path)) {
		std::cerr << "�޷���ȡ������ģ���ļ���ַ��" << std::endl;
		config_file.close();
		return -1;
	}
	if (!std::getline(config_file, hand_cfg.hamer_path)) {
		std::cerr << "�޷���ȡ�ֲ���̬����ģ���ļ���ַ��" << std::endl;
		config_file.close();
		return -1;
	}
	// �����Ϣ��ȡ
	if (!(config_file >> num_view)) {
		std::cerr << "�޷���ȡ��ͼ������" << std::endl;
		config_file.close();
		return -1;
	}
	// ���ж�ȡ����
	for (int i = 0; i < num_view; ++i) {
		// ��ʱ����洢һ��26��float
		float data[26];
		for (int j = 0; j < 26; ++j) {
			if (!(config_file >> data[j])) { // ��ȡʧ�ܴ���
				std::cerr << "�ļ���ʽ��������ݲ��㡣" << std::endl;
				config_file.close();
				return -1;
			}
		}
		// 1. �ڲξ��� 3x3 (ǰ9��Ԫ��)
		intr.emplace_back(cv::Mat(3, 3, CV_32F, data).clone());
		// 2. ��ת���� 3x3 (�м�9��Ԫ��)
		rot_trans.emplace_back(cv::Mat(3, 3, CV_32F, data + 9).clone());
		// 3. ƽ������ 3x1 (������3��Ԫ��)
		transl_trans.emplace_back(cv::Mat(3, 1, CV_32F, data + 18).clone());
		// 4. ������� 5�� (���5��Ԫ��)
		undist.emplace_back(std::vector<float> {data[21], data[22], data[23], data[24], data[25]});
	}
	config_file.close();

	// ******** ������̬������ ********
	BodyHand::PoseEstimator pe{
		body_cfg,
		hand_cfg,
		num_view,
		intr,
		rot_trans,
		transl_trans,
		undist
	};

	// ******** ���첶׽ϵͳ ********
	get_app();
	init_device();
	if (num_view != get_device_count()) {
		std::cerr << "�����ļ�����ͼ��ĿΪ��" << num_view << "��������������ĿΪ��" << get_device_count() << std::endl;
		close_device();
		return -1;
	}
	start_grabbing();
	
	// ******** ��ʼ�����ӻ����� ********
	window.setBackgroundColor(cv::viz::Color::white());
	window.spinOnce(1, true);

	// ******** ��׽����̬���� ********
	std::ofstream pose_ofs(get_time_string() + "_pose_output.txt"); // ���յ�ǰ������������̬����ļ�
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
				// ���ӻ���opencv
				updateViz(pose_result.body_kps_3d, pose_result.hand_kps_3d);
				// �洢���ļ���ÿ�е�һ����ʱ���
				pose_ofs << time_string << ' ';
				// Ȼ����ȫ��3d����ؽڵ�����
				for (const auto& kp : pose_result.body_kps_3d) {
					pose_ofs << kp.x << ' ' << kp.y << ' ' << kp.z << ' ';
				}
				// Ȼ����ȫ��42���ֲ��ؽڵ����꣬�������
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

	// ******** ֹͣ��׽���ͷ���Դ ********
	stop_grabbing();
	close_device();

	return 0;
}