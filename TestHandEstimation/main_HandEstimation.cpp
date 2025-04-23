#include <vector>
#include <opencv2/opencv.hpp>
#include "Pose3D.h"

int main() {
	BodyHand::BodyModelConfig body_cfg;
	body_cfg.model_path = "../BodyHand/models/yolov8s-pose.onnx";
	BodyHand::HandModelConfig hand_cfg;
	hand_cfg.handlr_path = "../BodyHand/models/handLR_480x640.onnx";
	hand_cfg.hamer_path = "../BodyHand/models/hand_mano.onnx";
	std::vector<cv::Mat> intr = { (cv::Mat_<float>(3, 3) << 1880, 0, 720, 0, 1880, 540, 0, 0, 1) };
	std::vector<cv::Mat> rot = { (cv::Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1) };
	std::vector<cv::Mat> transl = { (cv::Mat_<float>(3, 1) << 0, 0, 0) };
	std::vector<std::vector<float>> undist = { {0, 0, 0, 0, 0} };

	BodyHand::PoseEstimator pe{
		body_cfg,
		hand_cfg,
		1,
		intr,
		rot,
		transl,
		undist
	};

	cv::Mat img = cv::imread("wangsit.jpg");
	std::vector<cv::Point3f> kps_cam;
	std::vector<cv::Point2f> kps_img;
	auto [valid_left, valid_right] = pe.estimateHand(
		img,
		kps_cam,
		kps_img,
		0
	);

	if (valid_left) {
		for (int i = 0; i < 21; ++i) {
			cv::circle(img, kps_img[i], 3, cv::Scalar(0, 0, 255), -1);
		}
	}
	if (valid_right) {
		for (int i = 21; i < 42; ++i) {
			cv::circle(img, kps_img[i], 3, cv::Scalar(0, 255, 255), 1);
		}
	}

	cv::imshow("debug", img);
	cv::waitKey(0);

	return 0;
}