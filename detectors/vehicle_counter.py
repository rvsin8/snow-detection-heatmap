import cv2
import numpy as np
from collections import deque
import time


class VehicleCounter:
    """
    V1 vehicle counter:
      - Uses a background subtractor + contour area filtering
      - Counts an object when its centroid crosses a horizontal line (line_y)

    This is a pragmatic V1 (no YOLO). Works best with:
      - Fixed camera
      - Road roughly horizontal
      - line_y adjusted near where cars pass
    """

    def __init__(
        self,
        line_y: int | None,
        min_area: int = 2500,
        history: int = 300,
        var_threshold: int = 40,
        cooldown_sec: float = 0.6,
        window_sec: int = 60,
    ):
        self.line_y = line_y
        self.min_area = min_area
        self.cooldown_sec = cooldown_sec
        self.window_sec = window_sec

        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )

        self._last_count_time = 0.0
        self._recent_crossings = deque()  # timestamps of crossings (for per-minute rate)

    def set_line_y(self, line_y: int):
        self.line_y = line_y

    def process(self, frame):
        if self.line_y is None:
            raise ValueError("VehicleCounter.line_y is not set. Call set_line_y().")

        fg = self.back_sub.apply(frame)

        # Less sensitive than snow: threshold lower than 200, then close gaps
        _, mask = cv2.threshold(fg, 180, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crossings = 0
        debug_boxes = []

        # Find “vehicle-like” blobs
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2

            debug_boxes.append((x, y, w, h, cx, cy))

            # Simple crossing: centroid close to line (band)
            band = 8
            if abs(cy - self.line_y) <= band:
                now = time.time()
                if (now - self._last_count_time) >= self.cooldown_sec:
                    crossings += 1
                    self._last_count_time = now
                    self._recent_crossings.append(now)

        # Remove old crossings for per-minute rate
        now = time.time()
        while self._recent_crossings and (now - self._recent_crossings[0] > self.window_sec):
            self._recent_crossings.popleft()

        vehicles_per_min = len(self._recent_crossings)

        metrics = {
            "crossings_this_frame": crossings,
            "vehicles_per_min": vehicles_per_min,
        }

        debug = {
            "line_y": self.line_y,
            "boxes": debug_boxes,
        }

        return metrics, debug