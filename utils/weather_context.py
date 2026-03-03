import json
import os
import time


class WeatherContextLoader:
    """
    Reads a simple JSON file you update manually (from your weather app).
    This avoids APIs and still gives you live context for fusion.
    """
    def __init__(self, filepath="weather_context.json", reload_every_sec=30):
        self.filepath = filepath
        self.reload_every_sec = reload_every_sec
        self._last_load = 0.0
        self._cache = None
        self._last_mtime = None

    def get(self):
        now = time.time()
        if (now - self._last_load) < self.reload_every_sec:
            return self._cache

        self._last_load = now

        if not os.path.exists(self.filepath):
            self._cache = None
            return None

        mtime = os.path.getmtime(self.filepath)
        if self._last_mtime is not None and mtime == self._last_mtime:
            return self._cache

        self._last_mtime = mtime

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        except Exception:
            self._cache = None

        return self._cache


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _normalize_snow_rate_inhr(rate):
    """
    Map snow rate (in/hr) to 0..100.
    Typical ranges:
      0.0  -> 0
      0.5  -> 25
      1.0  -> 50
      2.0+ -> 100
    """
    if rate is None:
        return None
    try:
        r = float(rate)
    except Exception:
        return None
    return _clamp((r / 2.0) * 100.0, 0.0, 100.0)


def fuse_intensity(cv_intensity_pct, weather_context):
    """
    Hybrid intensity:
      60% CV
      40% Weather snow-rate (if available)
    If weather is missing, returns CV.
    """
    cv = _clamp(float(cv_intensity_pct), 0.0, 100.0)

    if not weather_context:
        return cv

    wr = _normalize_snow_rate_inhr(weather_context.get("snow_rate_in_per_hr"))
    if wr is None:
        return cv

    return 0.6 * cv + 0.4 * wr