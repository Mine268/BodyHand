from typing import Optional
from dataclasses import dataclass
import numpy as np
import mediapipe as mp


# base alias
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# hyper
HAND_THRESHOLD = 0.2
PRESENCE_THRESHOLD = 0.5
TRACKING_THRESHOLD = 0.5


@dataclass
class HandResult:
    leftHand: Optional[np.ndarray] = None
    rightHand: Optional[np.ndarray] = None
    leftHand_25D: Optional[np.ndarray] = None
    rightHand_25D: Optional[np.ndarray] = None
    leftHand_3D: Optional[np.ndarray] = None
    rightHand_3D: Optional[np.ndarray] = None


def listLmk2np(lmk_list) -> np.ndarray:
    lens = len(lmk_list)
    retval = np.zeros(shape=(lens, 3), dtype=np.float32)
    for i, lmk in enumerate(lmk_list):
        retval[i] = np.array([lmk.x, lmk.y, lmk.z])
    return retval


class HandDetector():
    def __init__(
        self,
        model_asset_path,
    ):
        self.model_path = model_asset_path

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=HAND_THRESHOLD,
            min_hand_presence_confidence=PRESENCE_THRESHOLD,
            min_tracking_confidence=TRACKING_THRESHOLD)

        self.landmarker = HandLandmarker.create_from_options(options)

    def __del__(self):
        self.landmarker.close()

    def estimate(self, img: np.ndarray, time_ms: int) -> HandResult:
        '''
        img: [3,H,W] rgb
        time_ms: timestamp in ms
        ret: [J,3]
        '''
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        hand_landmarker_result = self.landmarker.detect_for_video(
            mp_image, time_ms)

        handedness = hand_landmarker_result.handedness
        local_landmark = hand_landmarker_result.hand_landmarks
        cam_landmark = hand_landmarker_result.hand_world_landmarks

        for cat in handedness:
            cat.sort(key=lambda e: e.score, reverse=True)

        hand_index = {"Right": None, "Left": None}
        max_score = {"Right": -1., "Left": -1.}

        for ix, cat in enumerate(handedness):
            # cat[0] has highest score
            score = cat[0].score
            hand_type = cat[0].category_name

            if hand_index[hand_type] is None or max_score[hand_type] < score:
                max_score[hand_type] = score
                hand_index[hand_type] = ix

        height, width, _ = img.shape

        rightHand, rightHand_25D = None, None
        leftHand, leftHand_25D = None, None
        if hand_index["Right"] is not None:
            rightHand = listLmk2np(cam_landmark[hand_index["Right"]])
            rightHand_25D = listLmk2np(local_landmark[hand_index["Right"]])
            rightHand_25D[:, :2] *= np.array([[width, height]])
        if hand_index["Left"] is not None:
            leftHand = listLmk2np(cam_landmark[hand_index["Left"]])
            leftHand_25D = listLmk2np(local_landmark[hand_index["Left"]])
            leftHand_25D[:, :2] *= np.array([[width, height]])


        return HandResult(
            leftHand=leftHand, rightHand=rightHand,
            leftHand_25D=leftHand_25D, rightHand_25D=rightHand_25D)


def get_classes():
    return HandDetector, HandResult