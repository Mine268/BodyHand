#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

/**
 * @brief 从棋盘格图像估计相机相对于棋盘的位姿。
 *
 * @param image        输入图像（BGR 或灰度都可）
 * @param boardSize    棋盘内角点数量（列数、行数），如 cv::Size(11, 8)
 * @param squareSize   单格实际边长（单位自定，米/毫米皆可）
 * @param K            相机内参矩阵 3x3（CV_64F），如：
 *                     [ fx  0  cx
 *                       0  fy  cy
 *                       0   0   1 ]
 * @param distCoeffs   畸变系数（CV_64F），可为 1x5、1x8 等；若无畸变可传 5x1 全零
 * @param rvec         输出旋转向量（Rodrigues）
 * @param tvec         输出平移向量
 * @param draw         是否在图像上绘制检测到的角点与坐标轴
 * @param imageOut     若 draw=true，返回带可视化的图像（BGR）
 * @param reprojErr    可选输出：平均重投影误差（像素）
 * @return true        成功；false 失败（未找到角点或 PnP 失败）
 */
inline bool estimatePoseFromChessboard(const cv::Mat& image,
    const cv::Size& boardSize,
    double squareSize,
    const cv::Mat& K,
    const cv::Mat& distCoeffs,
    cv::Vec3d& rvec,
    cv::Vec3d& tvec,
    bool draw = false,
    cv::Mat* imageOut = nullptr,
    double* reprojErr = nullptr)
{
    // ---- 基本检查 ----
    if (image.empty() || boardSize.width < 2 || boardSize.height < 2 || squareSize <= 0.0) {
        std::cerr << "[estimatePoseFromChessboard] invalid input.\n";
        return false;
    }
    if (K.empty() || K.rows != 3 || K.cols != 3 || K.type() != CV_64F) {
        std::cerr << "[estimatePoseFromChessboard] camera matrix K must be 3x3 CV_64F.\n";
        return false;
    }
    if (distCoeffs.empty()) {
        std::cerr << "[estimatePoseFromChessboard] warning: distCoeffs is empty; assuming zero distortion.\n";
    }

    // ---- 生成棋盘 3D 物点（位于 z=0 的平面）----
    std::vector<cv::Point3d> objectPoints;
    objectPoints.reserve(static_cast<size_t>(boardSize.area()));
    for (int r = 0; r < boardSize.height; ++r) {
        for (int c = 0; c < boardSize.width; ++c) {
            objectPoints.emplace_back(c * squareSize, r * squareSize, 0.0);
        }
    }

    // ---- 提取角点 ----
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = image.clone();
    }

    std::vector<cv::Point2f> corners;
    const int findFlags = cv::CALIB_CB_ADAPTIVE_THRESH |
        cv::CALIB_CB_NORMALIZE_IMAGE |
        cv::CALIB_CB_FAST_CHECK; // 可按需移除 FAST_CHECK 以提高召回
    bool found = cv::findChessboardCorners(gray, boardSize, corners, findFlags);
    if (!found) {
        std::cerr << "[estimatePoseFromChessboard] chessboard not found.\n";
        return false;
    }

    // 亚像素精细化
    cv::cornerSubPix(
        gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01)
    );

    // ---- 选择求解器并求解位姿 ----
    int pnpFlag = 0;
#ifdef CV_SOLVEPNP_IPPE_SQUARE
    // 对平面、均匀网格更鲁棒
    pnpFlag = cv::SOLVEPNP_IPPE_SQUARE;
#else
    pnpFlag = cv::SOLVEPNP_ITERATIVE;
#endif

    bool ok = cv::solvePnP(objectPoints, corners, K, distCoeffs, rvec, tvec, false, pnpFlag);
    if (!ok) {
        std::cerr << "[estimatePoseFromChessboard] solvePnP failed.\n";
        return false;
    }

    // ---- 可选：计算重投影误差 ----
    if (reprojErr) {
        std::vector<cv::Point2f> reprojected;
        cv::projectPoints(objectPoints, rvec, tvec, K, distCoeffs, reprojected);
        double errSum = 0.0;
        for (size_t i = 0; i < reprojected.size(); ++i) {
            errSum += cv::norm(reprojected[i] - corners[i]);
        }
        *reprojErr = errSum / static_cast<double>(reprojected.size());
    }

    // ---- 可选：绘制结果 ----
    if (draw) {
        cv::Mat vis;
        if (image.channels() == 1)
            cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
        else
            vis = image.clone();

        // 绘制角点
        cv::drawChessboardCorners(vis, boardSize, corners, true);

        // 在棋盘原点处画坐标轴（长度 = 3 * squareSize）
        const float axisLen = static_cast<float>(3.0 * squareSize);
        std::vector<cv::Point3f> axisPts = {
            {0, 0, 0},
            {axisLen, 0, 0},
            {0, axisLen, 0},
            {0, 0, axisLen}
        };
        std::vector<cv::Point2f> axisImg;
        cv::projectPoints(axisPts, rvec, tvec, K, distCoeffs, axisImg);

        // 画三根轴
        cv::line(vis, axisImg[0], axisImg[1], cv::Scalar(0, 0, 255), 2);   // X - 红
        cv::line(vis, axisImg[0], axisImg[2], cv::Scalar(0, 255, 0), 2);   // Y - 绿
        cv::line(vis, axisImg[0], axisImg[3], cv::Scalar(255, 0, 0), 2);   // Z - 蓝

        if (imageOut) *imageOut = std::move(vis);
    }

    return true;
}

/**
 * @brief 将 (rvec, tvec) 转为 4x4 齐次位姿矩阵（double）
 *        T_cam_board：把棋盘系下点变换到相机系
 */
inline cv::Mat poseRtTo44(const cv::Vec3d& rvec, const cv::Vec3d& tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R); // 3x3, CV_64F
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    T.at<double>(0, 3) = tvec[0];
    T.at<double>(1, 3) = tvec[1];
    T.at<double>(2, 3) = tvec[2];
    return T;
}
