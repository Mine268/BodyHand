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

	struct PoseResult {
		bool valid_body, valid_left, valid_right;
		std::vector<cv::Point3f> body_kps_3d{};
		std::vector<std::vector<std::vector<cv::Point2f>>> body_kps_2d{};
		std::vector<std::vector<std::vector<float>>> body_kps_conf{};
		std::vector<std::vector<float>> body_conf{};
		std::vector<cv::Point3f> hand_kps_3d{};
		std::vector<cv::Point2f> hand_kps_2d{};
		std::vector<cv::Rect2f> hand_bbox{}; // xywh
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
		/// <returns>(������Ч��, ������Ч��)</returns>
		std::tuple<bool, bool> estimateHand(
			IN cv::Mat& img,
			OUT std::vector<cv::Point3f>& _kps_cam,
			OUT std::vector<cv::Point2f>& _kps_img,
			OUT std::optional<std::reference_wrapper<std::vector<cv::Rect2f>>> hand_bbox = std::nullopt,
			IN int view_ix = 0
		);

		/// <summary>
		/// ����ȫ�����̬����
		/// </summary>
		/// <param name="imgs">����ͼͼ��</param>
		/// <param name="body_kps">����ؽڵ�</param>
		/// <param name="hand_kps">�ֲ��ؽڵ㣬ǰ21�������ֵģ���21���������</param>
		/// <param name="hand_ref_view">����һ����ͼ�н����ֲ�����</param>
		/// <returns>(ȫ����Ч��, ������Ч��, ������Ч��)</returns>
		std::tuple<bool, bool, bool> estimatePose(
			IN std::vector<cv::Mat>& imgs,
			OUT std::vector<cv::Point3f>& body_kps,
			OUT std::vector<cv::Point3f>& hand_kps,
			IN int hand_ref_view = 0
		);

		void estimatePose(
			IN std::vector<cv::Mat>& imgs,
			OUT PoseResult& pose_result,
			IN int hand_ref_view = 0
		);

	private:
		bool loadBodyModel();
		bool loadHandModel();

		std::vector<cv::Point3f> triangulate2DPoints(
			const std::vector<std::vector<cv::Point2f>>& img_coords
		);

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