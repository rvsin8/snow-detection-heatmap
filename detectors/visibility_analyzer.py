import cv2
import numpy as np


class VisibilityAnalyzer:
    """
    V1 visibility:
      - Laplacian variance (blur indicator)
      - Edge density (contrast/edges indicator)

    Output:
      - visibility_score (0..100)
      - visibility_label: "Good" | "Moderate" | "Poor"
    """

    def __init__(self):
        pass

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur metric (higher = sharper)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())

        # Edge density
        edges = cv2.Canny(gray, 60, 140)
        edge_density = float(np.mean(edges > 0))  # 0..1

        # Heuristic normalize to 0..100
        # Typical lap_var ranges vary wildly by camera; we clamp to keep stable UI.
        lap_norm = min(1.0, lap_var / 800.0)  # tune if needed
        edge_norm = min(1.0, edge_density / 0.08)  # tune if needed

        score = (0.65 * lap_norm + 0.35 * edge_norm) * 100.0

        if score >= 65:
            label = "Good"
        elif score >= 40:
            label = "Moderate"
        else:
            label = "Poor"

        return {
            "visibility_score": score,
            "visibility_label": label,
            "lap_var": lap_var,
            "edge_density": edge_density,
        }