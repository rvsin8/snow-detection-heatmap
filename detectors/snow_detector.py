import cv2
import numpy as np


class SnowDetector:
    def __init__(self):
        self.prev_gray = None
        self.prev_intensity_pct = 0.0
        self.light_center = None

    # ==========================================
    # Find Brightest Light Source Automatically
    # ==========================================
    def detect_light_source(self, gray):
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
        return maxLoc

    # ==========================================
    # Create Smaller Circular ROI Around Light
    # ==========================================
    def create_light_mask(self, shape, center, radius=160):
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        return mask

    # ==========================================
    # Main Snow Process
    # ==========================================
    def process(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect brightest point (streetlight)
        self.light_center = self.detect_light_source(gray)

        # Smaller detection zone
        light_mask = self.create_light_mask(
            gray.shape,
            self.light_center,
            radius=160
        )

        masked_gray = cv2.bitwise_and(gray, gray, mask=light_mask)

        # Enhance contrast in light beam
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        enhanced_gray = clahe.apply(masked_gray)

        # Bright snow detection
        _, bright_mask = cv2.threshold(enhanced_gray, 190, 255, cv2.THRESH_BINARY)

        if self.prev_gray is None:
            self.prev_gray = enhanced_gray
            motion_mask = np.zeros_like(enhanced_gray)
        else:
            diff = cv2.absdiff(enhanced_gray, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            self.prev_gray = enhanced_gray

        combined = cv2.bitwise_and(bright_mask, motion_mask)
        combined = cv2.medianBlur(combined, 3)

        # Count snow pixels only inside beam
        snow_pixels = np.sum(combined > 0)
        total_pixels = np.sum(light_mask > 0)

        intensity_pct = (snow_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0

        rate = intensity_pct - self.prev_intensity_pct
        self.prev_intensity_pct = intensity_pct

        # Overlay visualization
        overlay = frame.copy()
        overlay[combined > 0] = [0, 255, 255]
        cv2.circle(overlay, self.light_center, 160, (0, 255, 0), 2)

        metrics = {
            "intensity_pct": intensity_pct,
            "rate_per_frame": rate,
        }

        return frame, overlay, metrics