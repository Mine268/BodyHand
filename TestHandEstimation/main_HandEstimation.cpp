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
	pe.estimateHand(
		img,
		kps_cam,
		kps_img,
		0
	);

	for (auto j2d : kps_img) {
		cv::circle(img, j2d, 3, cv::Scalar(0, 0, 255), -1);
	}
	cv::imshow("debug", img);
	cv::waitKey(0);

	return 0;
}