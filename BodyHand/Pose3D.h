#pragma once

#include <vector>
#include <string>
#include <optional>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "YoloONNX.h"
#include "HaMeRONNX.h"

#define IN
#define OUT

namespace BodyHand
{
	struct CameraParameter {
		cv::Mat intrinsics;
		std::vector<float> undist;
		cv::Mat rotation, translation;
		cv::Mat rot_transformation, transl_transformation;
	};

	struct BodyModelConfig {
		std::string model_path;
	};

	struct HandModelConfig {
		std::string detect_path;
		std::string lr_path;
		std::string hamer_path;
	};

	/// <summary>
	/// 手+身体关节点检测。需要标定的多视图。
	/// </summary>
	class PoseEstimator {
	public:
		PoseEstimator(
			BodyModelConfig _body_model_cfg,
			HandModelConfig _hand_model_cfg,
			int _num_views,
			const std::vector<cv::Mat>& _intrinsics,
			const std::vector<cv::Mat>& _rot_transformations,
			const std::vector<cv::Mat>& _transl_transformation,
			const std::vector<std::vector<float>>& _undists
		);
		~PoseEstimator() {}

		/// <summary>
		/// 估计图像中的二维人体姿态
		/// </summary>
		/// <param name="imgs">图像列表</param>
		/// <param name="kpss2d">每张图像中的所有人的人体姿态</param>
		/// <param name="conf_kpss">每个关节点的置信度</param>
		/// <param name="conf_bodies">每个人检测的置信度</param>
		/// <returns>返回结果是否有效</returns>
		bool estimateBody(
			IN std::vector<cv::Mat>& imgs,
			OUT std::vector<std::vector<std::vector<cv::Point2f>>>& kpss2d,
			OUT std::vector<std::vector<std::vector<float>>>& conf_kpss,
			OUT std::vector<std::vector<float>>& conf_bodies
		);

	private:
		bool loadBodyModel();
		bool loadHandModel();

	private:
		// 人体检测的 yolo 模型的地址和模型
		BodyModelConfig body_model_cfg;
		Yolov8Onnx body_model;
		// 人手检测的 HaMeR 模型的地址
		HandModelConfig hand_model_cfg;
		HaMeROnnx hand_model;
		// 使用的视图数量
		int num_views;
		// 相机的参数
		std::vector<CameraParameter> cameras;
	};

}