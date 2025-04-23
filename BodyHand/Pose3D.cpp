#include <exception>
#include <algorithm>
#include <opencv2/sfm/triangulation.hpp>
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
		// �������
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

		// ��������ڲ�
		for (std::size_t i = 0; i < _intrinsics.size(); ++i) {
			this->cameras.emplace_back(
				_intrinsics[i],
				_undists[i],
				_rot_transformations[i],
				_transl_transformation[i]
			);
		}

		// ����ģ��
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
		return hand_model.loadModel(hand_model_cfg.handlr_path, hand_model_cfg.hamer_path);
	}

	bool PoseEstimator::estimateBody(
		IN std::vector<cv::Mat>& imgs,
		OUT std::vector<std::vector<std::vector<cv::Point2f>>>& kpss2d,
		OUT std::vector<std::vector<std::vector<float>>>& conf_kpss,
		OUT std::vector<std::vector<float>>& conf_bodies
	) {
		kpss2d.clear();
		conf_kpss.clear();
		conf_bodies.clear();
		for (auto &img : imgs) {
			std::vector<OutputPose> poses_2d;
			if (!body_model.OnnxDetect(img, poses_2d)) {
				return false;
			}
			std::vector<std::vector<cv::Point2f>> view2d; // ÿ��ͼ���е����ж�ά��̬
			std::vector<std::vector<float>> conf_kps;
			std::vector<float> conf_body;
			for (auto &pose_2d : poses_2d) {
				std::vector<cv::Point2f> man2d; // �����˵Ķ�ά��̬
				std::vector<float> man_conf; // �����˵Ĺؽڵ����Ŷ�
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

	std::tuple<bool, bool> PoseEstimator::estimateHand(
		IN cv::Mat& img,
		OUT std::vector<cv::Point3f>& _kps_cam,
		OUT std::vector<cv::Point2f>& _kps_img,
		OUT std::optional<std::reference_wrapper<std::vector<cv::Rect2f>>> hand_bbox,
		IN int view_ix
	) {
		if (view_ix >= cameras.size()) {
			throw std::runtime_error("view_ix out of range");
		}

		auto [valid_left, valid_right] = this->hand_model.detectPose(
			img,
			cameras[view_ix].intrinsics,
			cameras[view_ix].undist,
			_kps_cam,
			_kps_img,
			hand_bbox
		);

		return { valid_left, valid_right };
	}
	
	std::vector<cv::Point3f> PoseEstimator::triangulate2DPoints(const std::vector<std::vector<cv::Point2f>>& img_coords) {
		// ����ͶӰ����
		std::vector<cv::Mat_<double>> projections_double;
		for (const auto& cam : cameras) {
			cv::Mat proj, proj_double;
			cv::hconcat(cam.rot_transformation, cam.transl_transformation, proj);
			proj = cam.intrinsics * proj;
			proj.convertTo(proj_double, CV_64F);
			projections_double.emplace_back(std::move(proj_double));
		}
		std::vector<cv::Mat_<double>> img_coords_double;
		img_coords_double.reserve(img_coords.size());
		for (const auto& coords : img_coords) {
			const int N = coords.size();
			cv::Mat_<double> mat(2, N);
			for (int i = 0; i < N; ++i) {
				mat.at<double>(0, i) = coords[i].x;
				mat.at<double>(1, i) = coords[i].y;
			}
			img_coords_double.emplace_back(std::move(mat));
		}
		cv::Mat kps_3d_double;
		cv::sfm::triangulatePoints(img_coords_double, projections_double, kps_3d_double);
		std::vector<cv::Point3f> kps_3d(kps_3d_double.cols);
		for (int i = 0; i < kps_3d_double.cols; ++i) {
			kps_3d[i] = cv::Point3f{
				static_cast<float>(kps_3d_double.at<double>(0, i)),
				static_cast<float>(kps_3d_double.at<double>(1, i)),
				static_cast<float>(kps_3d_double.at<double>(2, i))
			};
		}
		return kps_3d;
	}

	std::tuple<bool, bool, bool> PoseEstimator::estimatePose(
		IN std::vector<cv::Mat>& imgs,
		OUT std::vector<cv::Point3f>& body_kps,
		OUT std::vector<cv::Point3f>& hand_kps,
		IN int hand_ref_view
	) {
		if (hand_ref_view >= cameras.size()) {
			throw std::runtime_error("view_ix out of range");
		}

		hand_kps.clear();
		hand_kps.resize(42);

		// ������̬����
		std::vector<std::vector<std::vector<cv::Point2f>>> body_kps_2d;
		std::vector<std::vector<std::vector<float>>> body_kps_conf;
		std::vector<std::vector<float>> body_conf;
		bool valid_body = estimateBody(imgs, body_kps_2d, body_kps_conf, body_conf);

		// ÿ����ͼĬ��ѡ��0����
		std::vector<std::vector<cv::Point2f>> body_kps_2d_selected;
		for (const auto& k2d : body_kps_2d) {
			body_kps_2d_selected.emplace_back(k2d[0]);
		}

		// ����ͼ���ǻ�
		std::vector<cv::Point3f> body_kps_3d;
		body_kps_3d = triangulate2DPoints(body_kps_2d_selected);

		// �ֲ���̬����
		std::vector<cv::Point3f> hand_kps_3d;
		std::vector<cv::Point2f> hand_kps_2d;
		auto [valid_left, valid_right] = estimateHand(imgs[hand_ref_view], hand_kps_3d, hand_kps_2d, std::nullopt, hand_ref_view);

		body_kps = body_kps_3d;
		// ƴ��
		if (valid_left) {
			std::transform(hand_kps_3d.begin(), hand_kps_3d.begin() + 21, hand_kps.begin(),
				[&](const cv::Point3f& joint) {
					// ����ƴ�ӵ���������
					return joint - hand_kps_3d[0] + body_kps[9];
				}
			);
		}
		if (valid_right) {
			std::transform(hand_kps_3d.begin() + 21, hand_kps_3d.end(), hand_kps.begin() + 21,
				[&](const cv::Point3f& joint) {
					// ����ƴ�ӵ���������
					return joint - hand_kps_3d[21] + body_kps[10];
				}
			);
		}

		return { valid_body, valid_left, valid_right };
	}

	void PoseEstimator::estimatePose(
		IN std::vector<cv::Mat>& imgs,
		OUT PoseResult& pose_result,
		IN int hand_ref_view
	) {
		if (hand_ref_view >= cameras.size()) {
			throw std::runtime_error("view_ix out of range");
		}

		// ������̬����
		{
			std::vector<std::vector<std::vector<cv::Point2f>>> body_kps_2d;
			std::vector<std::vector<std::vector<float>>> body_kps_conf;
			std::vector<std::vector<float>> body_conf;
			bool valid_body = estimateBody(imgs, body_kps_2d, body_kps_conf, body_conf);

			// ÿ����ͼĬ��ѡ��0����
			std::vector<std::vector<cv::Point2f>> body_kps_2d_selected;
			for (const auto& k2d : body_kps_2d) {
				body_kps_2d_selected.emplace_back(k2d[0]);
			}

			// ����ͼ���ǻ�
			std::vector<cv::Point3f> body_kps_3d;
			body_kps_3d = triangulate2DPoints(body_kps_2d_selected);

			// �������
			pose_result.valid_body = valid_body;
			pose_result.body_kps_3d = std::move(body_kps_3d);
			pose_result.body_kps_2d = std::move(body_kps_2d);
			pose_result.body_kps_conf = std::move(body_kps_conf);
			pose_result.body_conf = std::move(body_conf);
		}

		// ������̬����
		{
			std::vector<cv::Point3f> hand_kps_3d;
			std::vector<cv::Point2f> hand_kps_2d;
			std::vector<cv::Rect2f> hand_bbox;
			auto [valid_left, valid_right] = estimateHand(
				imgs[hand_ref_view],
				hand_kps_3d,
				hand_kps_2d,
				hand_bbox,
				hand_ref_view
			);
			pose_result.hand_kps_3d = std::move(hand_kps_3d);
			pose_result.hand_kps_2d = std::move(hand_kps_2d);
			pose_result.hand_bbox = std::move(hand_bbox);
		}

		// ƴ��
		for (int i = 20; i >= 0; --i) {
			pose_result.hand_kps_3d[i] = pose_result.hand_kps_3d[i] - pose_result.hand_kps_3d[0] + pose_result.body_kps_3d[9];
			pose_result.hand_kps_3d[i + 21] = pose_result.hand_kps_3d[i + 21] - pose_result.hand_kps_3d[21] + pose_result.body_kps_3d[10];
		}

		return;
	}

}