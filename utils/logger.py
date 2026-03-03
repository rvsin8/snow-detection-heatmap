import os
import csv
import time


class MinuteCSVLogger:
    """
    Logs one row per minute to logs/blizzard_metrics.csv

    Columns:
      timestamp_epoch, snow_intensity_pct, snow_rate_per_frame, vehicles_per_min, visibility_label, visibility_score
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._last_logged_minute = None

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Ensure header exists
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_epoch",
                    "snow_intensity_pct",
                    "snow_rate_per_frame",
                    "vehicles_per_min",
                    "visibility_label",
                    "visibility_score",
                ])

    def maybe_log(self, snow_intensity, snow_rate, vehicles_per_min, visibility_label, visibility_score):
        now = time.time()
        minute_key = int(now // 60)

        if self._last_logged_minute == minute_key:
            return

        self._last_logged_minute = minute_key

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(now),
                round(float(snow_intensity), 3),
                round(float(snow_rate), 3),
                int(vehicles_per_min),
                str(visibility_label),
                round(float(visibility_score), 3),
            ])