import cv2
import numpy as np


class DashboardRenderer:

    def __init__(self):
        self.font_title = cv2.FONT_HERSHEY_SIMPLEX
        self.font_body = cv2.FONT_HERSHEY_SIMPLEX

    # ---------------- DARK PANEL ----------------
    def _dark_panel(self, img, x1, y1, x2, y2, alpha=0.8):
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 20), -1)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # ---------------- MAIN RENDER ----------------
    def render(
        self,
        base_frame,
        fps,
        snow_metrics,
        vehicle_metrics,
        visibility_metrics,
        vehicle_debug=None
    ):

        out = base_frame.copy()
        h, w = out.shape[:2]

        # =====================================================
        # TOP HEADER BAR
        # =====================================================
        out = self._dark_panel(out, 0, 0, w, 70, alpha=0.75)

        cv2.putText(
            out,
            "URBAN BLIZZARD MONITOR",
            (30, 42),
            self.font_title,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        status_line = (
            f"FPS {int(fps)}   |   "
            f"Snow {snow_metrics.get('intensity_pct', 0):.1f}%   |   "
            f"Vehicles {vehicle_metrics.get('vehicles_per_min', 0)}   |   "
            f"Visibility {visibility_metrics.get('visibility_label', 'N/A')}"
        )

        cv2.putText(
            out,
            status_line,
            (30, 63),
            self.font_body,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        # =====================================================
        # RIGHT METRICS PANEL
        # =====================================================
        panel_w = 360
        x1 = w - panel_w - 30
        y1 = 100
        x2 = w - 30
        y2 = y1 + 230

        out = self._dark_panel(out, x1, y1, x2, y2, alpha=0.85)

        lx = x1 + 25
        ly = y1 + 40
        gap = 50

        # Section Title
        cv2.putText(
            out,
            "LIVE METRICS",
            (lx, ly),
            self.font_title,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Snow Intensity (large focal metric)
        ly += gap

        snow_pct = snow_metrics.get('intensity_pct', 0)

        cv2.putText(
            out,
            f"{snow_pct:.1f}%",
            (lx, ly),
            self.font_title,
            1.6,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            out,
            "Snow Intensity",
            (lx + 170, ly - 8),
            self.font_body,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        # Vehicles
        ly += gap

        cv2.putText(
            out,
            f"{vehicle_metrics.get('vehicles_per_min', 0)} vehicles / min",
            (lx, ly),
            self.font_body,
            0.85,
            (230, 230, 230),
            2,
            cv2.LINE_AA
        )

        # Visibility
        ly += gap

        vis_label = visibility_metrics.get('visibility_label', 'N/A')
        vis_score = visibility_metrics.get('visibility_score', 0)

        cv2.putText(
            out,
            f"Visibility: {vis_label} ({vis_score:.0f})",
            (lx, ly),
            self.font_body,
            0.85,
            (230, 230, 230),
            2,
            cv2.LINE_AA
        )

        # =====================================================
        # VEHICLE DEBUG OVERLAY
        # =====================================================
        if vehicle_debug:

            line_y = vehicle_debug.get("line_y", None)
            boxes = vehicle_debug.get("boxes", [])

            # Counting line (soft gold)
            if line_y is not None:
                cv2.line(
                    out,
                    (0, line_y),
                    (w, line_y),
                    (255, 190, 0),
                    2,
                    cv2.LINE_AA
                )

            # Bounding boxes (softer green)
            for (x, y, bw, bh, cx, cy) in boxes[:10]:
                cv2.rectangle(
                    out,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 200, 120),
                    2
                )

        return out