#include <exception>
#include <algorithm>
#include "Pose3D.h"

namespace BodyHand {

	PoseEstimator::PoseEstimator(
		BodyModelConfig _body_model_cfg,
		HandModelConfig _hand_model_cfg,
		int _num_views,
		const std::vector<cv::Mat>& _intrinsics,
		const std::vector<cv::Mat>& _rot_transformations,
		const std::vector<cv::Mat>& _transl_transformation,
		const std::vector<std::vector<float>>& _undists
	) :
		body_model_cfg(_body_model_cfg),
		hand_model_cfg(_hand_model_cfg),
		num_views(_num_views)
	{
		// 参数检查
		if (!(
			this->num_views == _intrinsics.size() &&
			this->num_views == _rot_transformations.size() &&
			this->num_views == _transl_transformation.size() &&
			this->num_views == _undists.size()
			)) {
			throw std::runtime_error("Camera count doesn't match");
		}
		if (std::any_of(_undists.begin(), _undists.end(),
			[](auto &&innerVec) {
				return innerVec.size() != 5;
			})) {
			throw std::runtime_error("Expect #undistortion coef to be 5");
		}

		// 加载模型
		if (!loadBodyModel()) {
			throw std::runtime_error("Cannot load body model");
		}
		if (!loadHandModel()) {
			throw std::runtime_error("Cannot load hand model");
		}
	}

	bool PoseEstimator::loadBodyModel() {
		return body_model.ReadModel(body_model_cfg.model_path, false);
	}

	bool PoseEstimator::loadHandModel() {
		return true;
	}

	bool PoseEstimator::estimateBody(
		IN std::vector<cv::Mat>& imgs,
		OUT std::vector<std::vector<std::vector<cv::Point2f>>>& kpss2d,
		OUT std::vector<std::vector<std::vector<float>>>& conf_kpss,
		OUT std::vector<std::vector<float>>& conf_bodies
	) {
		for (auto &img : imgs) {
			std::vector<OutputPose> poses_2d;
			if (!body_model.OnnxDetect(img, poses_2d)) {
				return false;
			}
			std::vector<std::vector<cv::Point2f>> view2d; // 每个图像中的所有二维姿态
			std::vector<std::vector<float>> conf_kps;
			std::vector<float> conf_body;
			for (auto &pose_2d : poses_2d) {
				std::vector<cv::Point2f> man2d; // 单个人的二维姿态
				std::vector<float> man_conf; // 单个人的关节点置信度
				for (std::size_t j = 0; j < pose_2d.kps.size(); j += 3) {
					man2d.emplace_back(pose_2d.kps[j], pose_2d.kps[j + 1]);
					man_conf.emplace_back(pose_2d.kps[j + 2]);
				}
				view2d.emplace_back(std::move(man2d));
				conf_kps.emplace_back(std::move(man_conf));
				conf_body.emplace_back(pose_2d.confidence);
			}
			kpss2d.emplace_back(std::move(view2d));
			conf_kpss.emplace_back(std::move(conf_kps));
			conf_bodies.emplace_back(std::move(conf_body));
		}
		return true;
	}

}