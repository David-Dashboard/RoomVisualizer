"""Camera motion estimation between keyframes.

With metric depth available for every frame, relative pose reduces to a
well-conditioned 3D-to-2D problem: lift ORB matches from the previous frame
into 3D using its depth map, then solve PnP against their pixel locations in
the current frame.  That fixes both rotation and translation *in metres*, which
pure two-view epipolar geometry cannot do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import PipelineConfig
from ..types import CameraIntrinsics, DepthMap, Frame

log = logging.getLogger(__name__)


@dataclass
class PoseEstimate:
    transform: np.ndarray  # 4x4, maps previous-camera coords -> current-camera
    inliers: int
    matches: int
    ok: bool

    @property
    def prev_to_current(self) -> np.ndarray:
        return self.transform

    @property
    def current_to_prev(self) -> np.ndarray:
        return invert_rigid(self.transform)


def invert_rigid(transform: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform (cheaper and stabler than a solve)."""
    out = np.eye(4)
    rotation = transform[:3, :3]
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ transform[:3, 3]
    return out


class OrbOdometry:
    """Frame-to-frame relative pose from ORB features and metric depth."""

    def __init__(self, cfg: PipelineConfig, n_features: int = 3000):
        self.cfg = cfg
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _features(self, frame: Frame):
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        return self.orb.detectAndCompute(gray, None)

    def estimate(
        self,
        prev_frame: Frame,
        prev_depth: DepthMap,
        curr_frame: Frame,
        intr: CameraIntrinsics,
    ) -> PoseEstimate:
        kp_prev, desc_prev = self._features(prev_frame)
        kp_curr, desc_curr = self._features(curr_frame)
        identity = np.eye(4)

        if desc_prev is None or desc_curr is None or len(kp_prev) < 8 or len(kp_curr) < 8:
            log.debug("frame %d: too few features", curr_frame.index)
            return PoseEstimate(identity, 0, 0, False)

        # Lowe ratio test - far more reliable than raw nearest-neighbour on ORB.
        raw = self.matcher.knnMatch(desc_prev, desc_curr, k=2)
        matches = [
            pair[0]
            for pair in raw
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        if len(matches) < self.cfg.min_matches:
            log.debug("frame %d: %d matches < %d", curr_frame.index, len(matches), self.cfg.min_matches)
            return PoseEstimate(identity, 0, len(matches), False)

        depth = prev_depth.depth
        h, w = depth.shape
        object_points = []
        image_points = []
        for m in matches:
            u, v = kp_prev[m.queryIdx].pt
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < w and 0 <= vi < h):
                continue
            z = float(depth[vi, ui])
            if not np.isfinite(z) or z <= 0:
                continue
            x = (u - intr.cx) / intr.fx * z
            y = (v - intr.cy) / intr.fy * z
            object_points.append((x, y, z))
            image_points.append(kp_curr[m.trainIdx].pt)

        if len(object_points) < max(6, self.cfg.min_matches // 2):
            log.debug("frame %d: only %d matches had depth", curr_frame.index, len(object_points))
            return PoseEstimate(identity, 0, len(matches), False)

        obj = np.asarray(object_points, dtype=np.float64)
        img = np.asarray(image_points, dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            intr.matrix,
            None,
            reprojectionError=3.0,
            iterationsCount=500,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        n_inliers = 0 if inliers is None else int(len(inliers))
        if not ok or n_inliers < 6:
            log.debug("frame %d: PnP failed (%d inliers)", curr_frame.index, n_inliers)
            return PoseEstimate(identity, n_inliers, len(matches), False)

        # Refine on the inlier set only.
        if n_inliers >= 6:
            idx = inliers.reshape(-1)
            rvec, tvec = cv2.solvePnPRefineLM(
                obj[idx], img[idx], intr.matrix, None, rvec, tvec
            )

        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec.reshape(3)
        return PoseEstimate(transform, n_inliers, len(matches), True)


def estimate_trajectory(
    frames: list[Frame],
    depths: list[DepthMap],
    intr: CameraIntrinsics,
    cfg: PipelineConfig,
) -> list[np.ndarray]:
    """Chain relative poses into camera-to-world transforms for every frame.

    Frame 0 defines the world frame.  When a link cannot be estimated the
    previous pose is carried forward, which keeps the sequence usable instead
    of scattering later frames at random.
    """
    poses = [np.eye(4)]
    if len(frames) == 1 or not cfg.estimate_poses:
        return poses + [np.eye(4)] * (len(frames) - 1)

    odom = OrbOdometry(cfg)
    failures = 0
    for i in range(1, len(frames)):
        estimate = odom.estimate(frames[i - 1], depths[i - 1], frames[i], intr)
        if estimate.ok:
            # pose_i (cam_i -> world) = pose_{i-1} @ (cam_i -> cam_{i-1})
            poses.append(poses[-1] @ estimate.current_to_prev)
            log.debug(
                "frame %d pose: %d/%d inliers, translation %.3f m",
                i,
                estimate.inliers,
                estimate.matches,
                float(np.linalg.norm(estimate.transform[:3, 3])),
            )
        else:
            failures += 1
            poses.append(poses[-1].copy())

    if failures:
        log.warning(
            "%d/%d frame links had no reliable pose; those frames reuse the "
            "previous camera pose",
            failures,
            len(frames) - 1,
        )
    return poses
