#include <vector>
#include <opencv2/opencv.hpp>
#include "Pose3D.h"

int main() {
	BodyHand::BodyModelConfig body_cfg;
	body_cfg.model_path = "../BodyHand/models/yolov8s-pose.onnx";
	BodyHand::HandModelConfig hand_cfg;
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

	cv::Mat img = cv::imread("bus.jpg");
	std::vector<cv::Mat> imgs = { img };
	std::vector<std::vector<std::vector<cv::Point2f>>> keypoints_2d;
	std::vector<std::vector<std::vector<float>>> conf_kps;
	std::vector<std::vector<float>> conf_bodies;
	bool flag = pe.estimateBody(
		imgs,
		keypoints_2d,
		conf_kps,
		conf_bodies
	);

	for (int p = 0; p < keypoints_2d[0].size(); ++p) {
		for (int j = 0; j < keypoints_2d[0][p].size(); ++j) {
			if (conf_kps[0][p][j] > 0.7) {
				cv::circle(
					img,
					keypoints_2d[0][p][j],
					2,
					cv::Scalar(0, 0, 255),
					-1
				);
			}
		}
	}

	cv::imwrite("bus_annot.jpg", img);

	return 0;
}