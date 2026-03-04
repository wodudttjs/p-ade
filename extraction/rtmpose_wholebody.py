#!/usr/bin/env python3
"""
RTMPose WholeBody (DWPose) 추론 모듈 - ONNX Runtime GPU

133 keypoints 출력:
  - body:  17 keypoints (COCO format)
  - foot:   6 keypoints
  - face:  68 keypoints
  - left_hand:  21 keypoints
  - right_hand: 21 keypoints

SO-101 로봇팔에 필요한 것:
  - body keypoints: shoulder, elbow, wrist (관절 각도 계산)
  - hand keypoints: 21개 × 2 (그리퍼 열림/닫힘 추정)
"""

import cv2
import numpy as np
from pathlib import Path

# cuDNN DLL 사전 로딩 — torch가 있으면 import하여 CUDA/cuDNN 경로 등록
# (onnxruntime 공식 문서 권장: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#compatibility-with-pytorch)
try:
    import torch  # noqa: F401 — cuDNN 9.x DLL preload
except ImportError:
    pass

import onnxruntime as ort

# ============================================================================
# YOLOX 사람 검출기
# ============================================================================

class YOLOXDetector:
    """YOLOX ONNX 기반 사람 검출 (grid decoding 포함)"""

    # YOLOX anchor-free: 3 scales × stride
    STRIDES = [8, 16, 32]

    def __init__(self, model_path: str, device: str = "gpu"):
        providers = self._get_providers(device)
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
        self.input_size = (self.input_shape[2], self.input_shape[3])  # (H, W)

        # 그리드 사전 생성 (YOLOX raw 출력 디코딩용)
        self._grids, self._strides_arr = self._build_grids()

    def _build_grids(self):
        """YOLOX 앵커 프리 그리드 생성"""
        ih, iw = self.input_size
        grids = []
        strides_expanded = []
        for stride in self.STRIDES:
            grid_h, grid_w = ih // stride, iw // stride
            yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
            grid = np.stack([xv, yv], axis=2).reshape(-1, 2).astype(np.float32)
            grids.append(grid)
            strides_expanded.append(np.full(grid_h * grid_w, stride, dtype=np.float32))
        return np.concatenate(grids, axis=0), np.concatenate(strides_expanded)

    @staticmethod
    def _get_providers(device: str):
        if device == "gpu" and "CUDAExecutionProvider" in ort.get_available_providers():
            return [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]
        return ["CPUExecutionProvider"]

    def __call__(self, image: np.ndarray, score_thr: float = 0.3) -> list[np.ndarray]:
        """사람 바운딩 박스 반환: list of [x1, y1, x2, y2, score]"""
        h, w = image.shape[:2]
        input_img, ratio = self._preprocess(image)

        outputs = self.session.run(None, {self.input_name: input_img})
        dets = self._postprocess(outputs[0], ratio, score_thr, h, w)
        return dets

    def _preprocess(self, image: np.ndarray):
        ih, iw = self.input_size
        h, w = image.shape[:2]
        ratio = min(ih / h, iw / w)
        new_h, new_w = int(h * ratio), int(w * ratio)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        blob = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        return blob, ratio

    def _postprocess(self, output, ratio, score_thr, orig_h, orig_w):
        """
        YOLOX 후처리 — grid 기반 디코딩 + NMS

        output shape: [1, 8400, 85]
          - [:4]  = raw box offsets (cx_off, cy_off, log_w, log_h)
          - [4]   = objectness (이미 sigmoid)
          - [5:]  = class scores (이미 sigmoid, COCO 80 classes)
        """
        predictions = output[0]  # [8400, 85]

        boxes_raw = predictions[:, :4]
        objectness = predictions[:, 4]
        class_scores = predictions[:, 5:]

        # person class = 0 → 점수
        person_scores = objectness * class_scores[:, 0]
        mask = person_scores > score_thr

        if not np.any(mask):
            return []

        # YOLOX grid 디코딩: (grid + offset) * stride
        cx = (self._grids[:, 0] + boxes_raw[:, 0]) * self._strides_arr
        cy = (self._grids[:, 1] + boxes_raw[:, 1]) * self._strides_arr
        bw = np.exp(boxes_raw[:, 2]) * self._strides_arr
        bh = np.exp(boxes_raw[:, 3]) * self._strides_arr

        # 필터링
        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        scores = person_scores[mask]

        # center → corner, 입력 스케일 → 원본 스케일
        x1 = np.clip((cx - bw / 2) / ratio, 0, orig_w)
        y1 = np.clip((cy - bh / 2) / ratio, 0, orig_h)
        x2 = np.clip((cx + bw / 2) / ratio, 0, orig_w)
        y2 = np.clip((cy + bh / 2) / ratio, 0, orig_h)

        dets = np.stack([x1, y1, x2, y2, scores], axis=1)

        # NMS
        keep = self._nms(dets, 0.45)
        return [dets[keep]]

    @staticmethod
    def _nms(dets, thresh):
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= thresh)[0]
            order = order[inds + 1]
        return keep


# ============================================================================
# RTMPose WholeBody (DWPose) 추론
# ============================================================================

class RTMPoseWholeBody:
    """
    DWPose ONNX 모델 추론
    입력: 이미지 + 바운딩 박스
    출력: 133 keypoints (body 17 + foot 6 + face 68 + left_hand 21 + right_hand 21)
    """

    # Keypoint 인덱스 범위
    BODY_RANGE = (0, 17)
    FOOT_RANGE = (17, 23)
    FACE_RANGE = (23, 91)
    LHAND_RANGE = (91, 112)
    RHAND_RANGE = (112, 133)

    # SO-101에 필요한 body keypoint 인덱스 (COCO format)
    # 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist
    SO101_BODY_INDICES = [5, 6, 7, 8, 9, 10]

    def __init__(self, model_path: str, device: str = "gpu"):
        providers = YOLOXDetector._get_providers(device)
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
        self.input_size = (self.input_shape[3], self.input_shape[2])  # (W, H)

    def __call__(self, image: np.ndarray, bbox: np.ndarray) -> dict:
        """
        단일 사람에 대한 wholebody 키포인트 추론

        Args:
            image: BGR 이미지
            bbox: [x1, y1, x2, y2] 또는 [x1, y1, x2, y2, score]

        Returns:
            dict with:
                keypoints: [133, 2] (x, y in original image coords)
                scores: [133] confidence
                body: [17, 3] (x, y, conf)
                left_hand: [21, 3]
                right_hand: [21, 3]
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)

        # 바운딩 박스 확장 (20%)
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        scale = max(bw, bh) * 1.2
        x1_e = int(cx - scale / 2)
        y1_e = int(cy - scale / 2)
        x2_e = int(cx + scale / 2)
        y2_e = int(cy + scale / 2)

        # 패딩 처리
        h, w = image.shape[:2]
        pad_left = max(0, -x1_e)
        pad_top = max(0, -y1_e)
        pad_right = max(0, x2_e - w)
        pad_bottom = max(0, y2_e - h)

        x1_c = max(0, x1_e)
        y1_c = max(0, y1_e)
        x2_c = min(w, x2_e)
        y2_c = min(h, y2_e)

        crop = image[y1_c:y2_c, x1_c:x2_c]
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 리사이즈 & 전처리
        inp_w, inp_h = self.input_size
        resized = cv2.resize(crop, (inp_w, inp_h))

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        blob = ((resized.astype(np.float32) - mean) / std).transpose(2, 0, 1)[np.newaxis]

        # 추론
        outputs = self.session.run(None, {self.input_name: blob})
        # simcc outputs: [1, 133, W*2], [1, 133, H*2]
        simcc_x = outputs[0][0]  # [133, W*2]
        simcc_y = outputs[1][0]  # [133, H*2]

        # argmax → 좌표 변환
        x_locs = np.argmax(simcc_x, axis=1).astype(np.float32)
        y_locs = np.argmax(simcc_y, axis=1).astype(np.float32)

        # confidence = geometric mean of max softmax, clamped to [0,1]
        x_scores = np.clip(np.max(simcc_x, axis=1), 0, 1)
        y_scores = np.clip(np.max(simcc_y, axis=1), 0, 1)
        scores = np.sqrt(x_scores * y_scores)

        # SimCC 좌표 → expanded bbox 좌표 → 원본 이미지 좌표
        # SimCC는 2x resolution (W*2, H*2)
        x_coords = x_locs / simcc_x.shape[1] * (x2_e - x1_e) + x1_e
        y_coords = y_locs / simcc_y.shape[1] * (y2_e - y1_e) + y1_e

        # 결과 파싱
        keypoints = np.stack([x_coords, y_coords], axis=1)  # [133, 2]

        def _slice_kps(start, end):
            kps = keypoints[start:end]
            sc = scores[start:end]
            return np.concatenate([kps, sc[:, np.newaxis]], axis=1)  # [N, 3]

        return {
            "keypoints": keypoints,    # [133, 2]
            "scores": scores,          # [133]
            "body": _slice_kps(*self.BODY_RANGE),         # [17, 3]
            "left_hand": _slice_kps(*self.LHAND_RANGE),   # [21, 3]
            "right_hand": _slice_kps(*self.RHAND_RANGE),  # [21, 3]
            "face": _slice_kps(*self.FACE_RANGE),         # [68, 3]
        }


# ============================================================================
# 비디오 포즈 추출 파이프라인 (MediaPipe 대체)
# ============================================================================

class RTMPoseVideoExtractor:
    """
    RTMPose WholeBody로 비디오에서 포즈+손 추출
    MediaPipe extract_pose_from_video()의 드롭인 대체
    """

    def __init__(self, device: str = "gpu"):
        model_dir = Path(__file__).parent.parent / "models" / "rtmpose"
        det_path = str(model_dir / "yolox_l.onnx")
        pose_path = str(model_dir / "dwpose_wholebody.onnx")

        if not Path(det_path).exists() or not Path(pose_path).exists():
            raise FileNotFoundError(
                f"RTMPose 모델 없음. 다음 경로에 모델을 배치하세요:\n"
                f"  {det_path}\n  {pose_path}"
            )

        print(f"🔧 RTMPose WholeBody 초기화 (device={device})...")
        self.detector = YOLOXDetector(det_path, device=device)
        self.pose_model = RTMPoseWholeBody(pose_path, device=device)

        provider = self.pose_model.session.get_providers()[0]
        print(f"  ✅ Provider: {provider}")

    def extract(self, video_path: str, output_fps: float = 5.0,
                max_frames: int = None) -> dict | None:
        """
        비디오에서 포즈 추출 (MediaPipe 호환 출력 포맷)

        Returns:
            dict with:
                body: [T, 17, 3]  (x_norm, y_norm, confidence)
                body_world: [T, 17, 3]  (same as body, RTMPose는 2D만)
                left_hand: [T, 21, 3]
                right_hand: [T, 21, 3]
                timestamps: [T]
                confidence: [T]
                fps: float
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(orig_fps / output_fps))

        body_list = []
        left_hand_list = []
        right_hand_list = []
        timestamps = []
        confidences = []

        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            if max_frames and processed >= max_frames:
                break

            h, w = frame.shape[:2]

            # 1) 사람 검출
            dets = self.detector(frame, score_thr=0.3)

            if not dets or len(dets[0]) == 0:
                # 검출 실패 → 전체 이미지를 바운딩 박스로 사용
                best_bbox = np.array([0, 0, w, h, 1.0])
            else:
                # 가장 큰 (또는 가장 신뢰도 높은) 사람 선택
                all_dets = dets[0]
                areas = (all_dets[:, 2] - all_dets[:, 0]) * (all_dets[:, 3] - all_dets[:, 1])
                best_idx = np.argmax(areas)
                best_bbox = all_dets[best_idx]

            # 2) 포즈 추론
            try:
                result = self.pose_model(frame, best_bbox)
            except Exception:
                frame_idx += 1
                continue

            # 3) 좌표 정규화 (0~1) — MediaPipe 호환
            body_kps = result["body"].copy()      # [17, 3] (x_pixel, y_pixel, conf)
            body_kps[:, 0] /= w
            body_kps[:, 1] /= h

            lh_kps = result["left_hand"].copy()    # [21, 3]
            lh_kps[:, 0] /= w
            lh_kps[:, 1] /= h

            rh_kps = result["right_hand"].copy()   # [21, 3]
            rh_kps[:, 0] /= w
            rh_kps[:, 1] /= h

            # 전체 포즈 신뢰도
            body_conf = np.mean(body_kps[:, 2])

            body_list.append(body_kps)
            left_hand_list.append(lh_kps)
            right_hand_list.append(rh_kps)
            timestamps.append(frame_idx / orig_fps)
            confidences.append(float(body_conf))

            frame_idx += 1
            processed += 1

        cap.release()

        if not body_list:
            return None

        return {
            "body": np.array(body_list, dtype=np.float32),           # [T, 17, 3]
            "body_world": np.array(body_list, dtype=np.float32),     # [T, 17, 3] (2D, 동일)
            "left_hand": np.array(left_hand_list, dtype=np.float32), # [T, 21, 3]
            "right_hand": np.array(right_hand_list, dtype=np.float32), # [T, 21, 3]
            "timestamps": np.array(timestamps, dtype=np.float32),
            "confidence": np.array(confidences, dtype=np.float32),
            "fps": output_fps,
            "num_body_keypoints": 17,
        }


# ============================================================================
# 편의 함수 (MediaPipe API 호환)
# ============================================================================

_extractor_instance = None

def extract_pose_from_video(video_path: str, output_fps: float = 5.0,
                            max_frames: int = None) -> dict | None:
    """
    MediaPipe extract_pose_from_video() 대체 — RTMPose WholeBody GPU 사용

    동일한 dict 형식을 반환하므로 기존 인코딩 함수와 호환됩니다.
    body shape만 [T, 33, 3] → [T, 17, 3]으로 변경됩니다.
    """
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = RTMPoseVideoExtractor(device="gpu")
    return _extractor_instance.extract(video_path, output_fps, max_frames)
