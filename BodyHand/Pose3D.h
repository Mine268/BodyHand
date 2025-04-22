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
		cv::Mat rot_transformation;
		cv::Mat transl_transformation;
	};

	struct BodyModelConfig {
		std::string model_path;
	};

	struct HandModelConfig {
		std::string handlr_path;
		std::string hamer_path;
	};

	/// <summary>
	/// ��+����ؽڵ��⡣��Ҫ�궨�Ķ���ͼ��
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
		/// ����ͼ���еĶ�ά������̬
		/// </summary>
		/// <param name="imgs">ͼ���б�</param>
		/// <param name="kpss2d">ÿ��ͼ���е������˵�������̬</param>
		/// <param name="conf_kpss">ÿ���ؽڵ�����Ŷ�</param>
		/// <param name="conf_bodies">ÿ���˼������Ŷ�</param>
		/// <returns>���ؽ���Ƿ���Ч</returns>
		bool estimateBody(
			IN std::vector<cv::Mat>& imgs,
			OUT std::vector<std::vector<std::vector<cv::Point2f>>>& kpss2d,
			OUT std::vector<std::vector<std::vector<float>>>& conf_kpss,
			OUT std::vector<std::vector<float>>& conf_bodies
		);

		/// <summary>
		/// �����ֲ���̬���ƣ�������һ�����ֺ�����
		/// </summary>
		/// <param name="img">���Ŵ�����ͼ��</param>
		/// <param name="_kps_cam">����42���ֱ�������+����������ռ��е�λ��</param>
		/// <param name="_kps_img">ͼ���ϵĹؼ���λ��</param>
		/// <param name="view_ix">����һ����ͼ�н��й���</param>
		/// <returns>ʼ��true</returns>
		bool estimateHand(
			IN cv::Mat& img,
			OUT std::vector<cv::Point3f>& _kps_cam,
			OUT std::vector<cv::Point2f>& _kps_img,
			IN int view_ix = 0
		);

		bool estimatePose(
			IN std::vector<cv::Mat>& imgs,
			OUT std::vector<cv::Point3f>& body_kps,
			OUT std::vector<cv::Point3f>& hand_kps,
			IN int hand_ref_view = 0
		);

	private:
		bool loadBodyModel();
		bool loadHandModel();

	private:
		// ������� yolo ģ�͵ĵ�ַ��ģ��
		BodyModelConfig body_model_cfg;
		Yolov8Onnx body_model;
		// ���ּ��� HaMeR ģ�͵ĵ�ַ
		HandModelConfig hand_model_cfg;
		HaMeROnnx hand_model;
		// ʹ�õ���ͼ����
		int num_views;
		// ����Ĳ���
		std::vector<CameraParameter> cameras;
	};

}