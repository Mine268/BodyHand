import cv2
import numpy as np
from HandDetector import HandDetector, HandResult  # 假设HandDetector类保存在hand_detector.py文件中

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 加载手部检测模型
    model_path = "hand_landmarker.task"  # 替换为你的模型路径
    hand_detector = HandDetector(model_asset_path=model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break

        # 转换为RGB格式（Mediapipe需要RGB格式）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 获取当前时间戳（毫秒）
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        # 检测手部
        hand_result = hand_detector.estimate(rgb_frame, timestamp_ms)

        # 绘制2D手部关键点
        if hand_result.leftHand_25D is not None:
            for landmark in hand_result.leftHand_25D:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # 黄色点

        if hand_result.rightHand_25D is not None:
            for landmark in hand_result.rightHand_25D:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # 红色点

        # 显示结果
        cv2.imshow('Hand Detection', frame)

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()