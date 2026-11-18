"""
LPU Solar-Glacial AI Computer Vision HUD Visualizer.
Renders futuristic biometric HUD scanning overlays, targeting reticles,
facial landmark constellation meshes, and Lovely Professional University branded identification badges.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from core.detector import DetectedFace
from core.matcher import MatchResult


class GlacialVisualizer:
    """
    Renders high-tech LPU solar-glacial AI vision HUD overlays on detected faces.
    Features:
    - LPU Solar Orange & Glacial Cyan targeting brackets
    - Facial landmark constellation lines (neural graph visualization)
    - Holographic identification tags with confidence bars
    - Anti-spoofing warning shields
    """

    # LPU Solar-Glacial BGR Palette
    LPU_ORANGE = (35, 112, 243)     # #f37023 (LPU Signature Solar Orange)
    CYAN_GLOW = (254, 242, 0)       # #00f2fe (Glacial Cyan)
    NEON_EMERALD = (125, 239, 56)   # #38ef7d (Aurora Green)
    FROST_WHITE = (255, 255, 255)
    SPOOF_RED = (80, 80, 255)       # Crimson Alert
    UNKNOWN_AMBER = (50, 165, 255)  # Amber

    @classmethod
    def draw_corner_brackets(
        cls,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        corner_len: int = 18,
        thickness: int = 2,
    ):
        """Draws high-tech sci-fi targeting brackets on face corners."""
        x, y, w, h = box
        # Top-Left
        cv2.line(image, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(image, (x, y), (x, y + corner_len), color, thickness)
        # Top-Right
        cv2.line(image, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(image, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(image, (x + w, y), (x + w, y + corner_len), color, thickness)
        # Bottom-Left
        cv2.line(image, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(image, (x, y + h), (x, y + h - corner_len), color, thickness)
        # Bottom-Right
        cv2.line(image, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

    @classmethod
    def draw_landmark_constellation(cls, image: np.ndarray, landmarks: np.ndarray, color: Tuple[int, int, int]):
        """
        Renders a glowing neural constellation mesh connecting facial landmarks:
        Right Eye <-> Left Eye <-> Nose <-> Mouth Corners.
        """
        pts = landmarks.astype(int)
        re, le, nose, rcm, lcm = pts[0], pts[1], pts[2], pts[3], pts[4]

        # Draw translucent neural connection lines
        overlay = image.copy()
        cv2.line(overlay, tuple(re), tuple(le), cls.CYAN_GLOW, 1, cv2.LINE_AA)
        cv2.line(overlay, tuple(re), tuple(nose), cls.CYAN_GLOW, 1, cv2.LINE_AA)
        cv2.line(overlay, tuple(le), tuple(nose), cls.CYAN_GLOW, 1, cv2.LINE_AA)
        cv2.line(overlay, tuple(nose), tuple(rcm), cls.CYAN_GLOW, 1, cv2.LINE_AA)
        cv2.line(overlay, tuple(nose), tuple(lcm), cls.CYAN_GLOW, 1, cv2.LINE_AA)
        cv2.line(overlay, tuple(rcm), tuple(lcm), cls.CYAN_GLOW, 1, cv2.LINE_AA)

        # Alpha blend lines for glow effect
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

        # Glowing landmark nodes
        for pt in pts:
            cv2.circle(image, tuple(pt), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, tuple(pt), 5, cls.LPU_ORANGE, 1, cv2.LINE_AA)

    @classmethod
    def render_hud_overlay(
        cls,
        image: np.ndarray,
        face: DetectedFace,
        match: MatchResult,
        is_live: bool = True,
        subject: str = "General",
        period: str = "Period 1",
    ) -> np.ndarray:
        """
        Renders a complete realistic sci-fi glacial AI HUD tag on the detected face.
        """
        h_img, w_img = image.shape[:2]
        x, y, w, h = face.box

        # Theme color choice
        if not is_live:
            theme_color = cls.SPOOF_RED
            status_text = "SPOOF ATTACK DETECTED"
        elif match.is_match:
            theme_color = cls.NEON_EMERALD
            status_text = f"LPU VERIFIED: {match.name}"
        else:
            theme_color = cls.LPU_ORANGE
            status_text = "LPU UNREGISTERED VISITOR"

        # 1. Subtle bounding box with outer LPU targeting brackets
        cv2.rectangle(image, (x, y), (x + w, y + h), theme_color, 1)
        cls.draw_corner_brackets(image, (x, y, w, h), theme_color, corner_len=min(24, w // 4), thickness=2)

        # 2. Draw Landmark Constellation Mesh
        cls.draw_landmark_constellation(image, face.landmarks, theme_color)

        # 3. Holographic Tech Tag Header
        hud_w = max(260, w + 40)
        hud_h = 60
        hud_x = max(8, min(x - 10, w_img - hud_w - 8))
        hud_y = y - hud_h - 10

        # If too close to top edge, flip below face
        if hud_y < 10:
            hud_y = y + h + 12

        # Draw translucent glacial backdrop for HUD badge
        sub_img = image[hud_y:hud_y + hud_h, hud_x:hud_x + hud_w]
        if sub_img.shape[0] == hud_h and sub_img.shape[1] == hud_w:
            glass_rect = np.full_like(sub_img, (18, 12, 6))
            cv2.addWeighted(glass_rect, 0.80, sub_img, 0.20, 0, sub_img)
            image[hud_y:hud_y + hud_h, hud_x:hud_x + hud_w] = sub_img

        # Border around HUD tag
        cv2.rectangle(image, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), theme_color, 1)

        # Top line: Status & Identity
        title = f"[LPU AI] {status_text}"
        cv2.putText(image, title, (hud_x + 8, hud_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, cls.FROST_WHITE, 1, cv2.LINE_AA)

        # Middle line: Latent Vector & Match Confidence
        if match.is_match:
            detail = f"ROLL/EMP: {match.user_id} | SIM: {match.confidence:.1f}% | 128D"
        else:
            detail = f"CONF: {match.confidence:.1f}% | ACCESS RESTRICTED"
        cv2.putText(image, detail, (hud_x + 8, hud_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, theme_color, 1, cv2.LINE_AA)

        # Bottom line: Period & Subject context
        subj_tag = f"CLASS: {period} - {subject[:20]}"
        cv2.putText(image, subj_tag, (hud_x + 8, hud_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 220, 240), 1, cv2.LINE_AA)

        # Micro confidence meter bar on bottom of badge
        conf_ratio = match.confidence / 100.0
        bar_w = int((hud_w - 16) * conf_ratio)
        cv2.rectangle(image, (hud_x + 8, hud_y + hud_h - 4), (hud_x + 8 + bar_w, hud_y + hud_h - 2), theme_color, -1)

        return image
