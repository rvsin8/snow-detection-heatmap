import cv2
import time

from ui.dashboard import DashboardRenderer
from detectors.snow_detector import SnowDetector
from detectors.vehicle_counter import VehicleCounter
from detectors.visibility_analyzer import VisibilityAnalyzer


def main():

    # ================= CAMERA =================
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # ================= SYSTEM COMPONENTS =================
    ui = DashboardRenderer()
    snow_detector = SnowDetector()
    visibility_analyzer = VisibilityAnalyzer()

    vehicle_counter = None  # will initialize after first frame

    prev_time = time.time()

    print("Press 'q' to quit.")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Initialize vehicle counter once
        if vehicle_counter is None:
            vehicle_counter = VehicleCounter(line_y=int(h * 0.6))

        # ================= FPS =================
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # ================= DETECTORS =================

        # SnowDetector returns: enhanced_frame, snow_overlay, snow_metrics
        enhanced_frame, snow_overlay, snow_metrics = snow_detector.process(frame)

        # VehicleCounter returns: vehicle_metrics, vehicle_debug
        vehicle_metrics, vehicle_debug = vehicle_counter.process(enhanced_frame)

        # VisibilityAnalyzer returns: visibility_metrics
        visibility_metrics = visibility_analyzer.process(enhanced_frame)

        # ================= RENDER DASHBOARD =================
        output = ui.render(
            base_frame=snow_overlay,
            fps=fps,
            snow_metrics=snow_metrics,
            vehicle_metrics=vehicle_metrics,
            visibility_metrics=visibility_metrics,
            vehicle_debug=vehicle_debug
        )

        cv2.imshow("Urban Blizzard Monitor", output)

        # ================= EXIT =================
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()