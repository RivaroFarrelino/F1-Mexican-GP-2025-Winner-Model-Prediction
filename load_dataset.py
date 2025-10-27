import os
import time
import json
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
from dateutil import parser as dtp

warnings.filterwarnings("ignore")

BASE = "https://api.openf1.org/v1"

def http_get(endpoint: str, params: Optional[Dict[str, Any]] = None, retries: int = 3, pause: float = 1.0):
    url = f"{BASE}/{endpoint}"
    last_exc = None
    for i in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.ok:
                return resp.json()
            last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            last_exc = e
        time.sleep(pause * (i + 1))
    if last_exc:
        raise last_exc
    return []

def ensure_datetime_utc(s: pd.Series, colname="date"):
    out = pd.to_datetime(s, utc=True, errors="coerce")
    if out.isna().all():
        return s.apply(lambda x: pd.NaT if pd.isna(x) else pd.to_datetime(x, utc=True, errors="coerce"))
    return out

def safe_mean(x):
    try:
        return float(pd.to_numeric(x, errors="coerce").dropna().mean())
    except Exception:
        return math.nan

def safe_min(x):
    try:
        return float(pd.to_numeric(x, errors="coerce").dropna().min())
    except Exception:
        return math.nan

def safe_sum(x):
    try:
        return float(pd.to_numeric(x, errors="coerce").dropna().sum())
    except Exception:
        return math.nan

@dataclass
class FetchConfig:
    outdir: str = "openf1_2025_until_austin"
    download_heavy: bool = True  
    car_speed_filter: int = 300     

class OpenF1Fetcher:
    def __init__(self, cfg: FetchConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.outdir, exist_ok=True)

    def _fetch_concat(self, endpoint: str, key_values: List[int], key_field: str = "session_key",
                      extra_params: Optional[Dict[str, Any]] = None, sleep: float = 0.25):
        frames = []
        for sk in key_values:
            params = {key_field: sk}
            if extra_params:
                params.update(extra_params)
            data = http_get(endpoint, params=params)
            if data:
                df = pd.DataFrame(data)
                df[key_field] = sk
                frames.append(df)
            time.sleep(sleep)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_all_until_austin(self):
        sessions = pd.DataFrame(http_get("sessions", {"year": 2025}))
        if sessions.empty:
            raise SystemExit("Tidak menemukan data sessions untuk 2025.")

        for c in ("date_start", "date_end"):
            if c in sessions.columns:
                sessions[c] = ensure_datetime_utc(sessions[c], c)

        mask_austin = sessions.get("location", pd.Series([None]*len(sessions))).astype(str).str.contains("Austin", case=False, na=False)
        mask_race   = sessions.get("session_name", pd.Series([None]*len(sessions))).astype(str).str.fullmatch("Race", case=False, na=False)
        austin_rows = sessions[mask_austin & mask_race].sort_values("date_start")

        if austin_rows.empty:
            mask_us = sessions.get("location", pd.Series([None]*len(sessions))).astype(str).str.contains("United States|USA|Austin", case=False, na=False)
            austin_rows = sessions[mask_us & mask_race].sort_values("date_start")

        if austin_rows.empty:
            raise SystemExit("Race Austin 2025 tidak ditemukan di endpoint sessions.")

        cutoff_date = austin_rows.iloc[-1]["date_start"]
        print(f"Cutoff Austin (UTC): {cutoff_date}")

        race_sessions = sessions[(mask_race) & (sessions["date_start"] <= cutoff_date)].copy()
        race_sessions = race_sessions.sort_values("date_start")
        race_sessions.to_csv(os.path.join(self.cfg.outdir, "sessions_race_upto_austin.csv"), index=False)

        session_keys = race_sessions["session_key"].dropna().astype(int).tolist()
        print(f"Total race sessions sampai Austin: {len(session_keys)}")

        df_results = self._fetch_concat("session_result", session_keys)
        df_grid    = self._fetch_concat("starting_grid", session_keys)
        df_drivers = self._fetch_concat("drivers", session_keys)

        if not df_results.empty:
            df_results.to_csv(os.path.join(self.cfg.outdir, "results_upto_austin.csv"), index=False)
        if not df_grid.empty:
            df_grid.to_csv(os.path.join(self.cfg.outdir, "starting_grid_upto_austin.csv"), index=False)
        if not df_drivers.empty:
            df_drivers.to_csv(os.path.join(self.cfg.outdir, "drivers_upto_austin.csv"), index=False)

        if self.cfg.download_heavy:
            df_laps   = self._fetch_concat("laps",   session_keys, sleep=0.35)
            df_pit    = self._fetch_concat("pit",    session_keys, sleep=0.35)
            df_stints = self._fetch_concat("stints", session_keys, sleep=0.35)
            df_wx     = self._fetch_concat("weather",session_keys, sleep=0.35)

            frames = []
            for sk in session_keys:
                d = http_get("car_data", {"session_key": sk, "speed>=": self.cfg.car_speed_filter})
                if d:
                    df = pd.DataFrame(d); df["session_key"] = sk
                    frames.append(df)
                time.sleep(0.4)
            df_car = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            if not df_laps.empty:   df_laps.to_csv(os.path.join(self.cfg.outdir, "laps_upto_austin.csv"), index=False)
            if not df_pit.empty:    df_pit.to_csv(os.path.join(self.cfg.outdir, "pit_upto_austin.csv"), index=False)
            if not df_stints.empty: df_stints.to_csv(os.path.join(self.cfg.outdir, "stints_upto_austin.csv"), index=False)
            if not df_wx.empty:     df_wx.to_csv(os.path.join(self.cfg.outdir, "weather_upto_austin.csv"), index=False)
            if not df_car.empty:    df_car.to_csv(os.path.join(self.cfg.outdir, f"car_data_speed{self.cfg.car_speed_filter}_upto_austin.csv"), index=False)

        return {
            "sessions": race_sessions,
            "results": df_results if 'df_results' in locals() else pd.DataFrame(),
            "grid": df_grid if 'df_grid' in locals() else pd.DataFrame(),
            "drivers": df_drivers if 'df_drivers' in locals() else pd.DataFrame(),
            "cutoff_utc": cutoff_date
        }

@dataclass
class FeatureConfig:
    out_features_csv: str = "features_up_to_austin_for_mexico.csv"
    track_altitude_m: float = 2285.0   # Mexico City
    track_length_km: float = 4.304
    overtake_difficulty: float = 0.65
    track_type: str = "high-altitude, medium-downforce"

class FeatureBuilder:
    def __init__(self, outdir: str, cfg: FeatureConfig, cutoff_utc):
        self.outdir = outdir
        self.cfg = cfg
        self.cutoff = cutoff_utc

    def _load_or_empty(self, name: str):
        path = os.path.join(self.outdir, name)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def build(self):
        results = self._load_or_empty("results_upto_austin.csv")
        grid    = self._load_or_empty("starting_grid_upto_austin.csv")
        drivers = self._load_or_empty("drivers_upto_austin.csv")
        laps    = self._load_or_empty("laps_upto_austin.csv")
        weather = self._load_or_empty("weather_upto_austin.csv")
        sessions = self._load_or_empty("sessions_race_upto_austin.csv")

        if results.empty:
            raise SystemExit("results_upto_austin.csv kosong — jalankan fetch dulu.")
        if sessions.empty:
            raise SystemExit("sessions_race_upto_austin.csv kosong — pastikan fetcher menyimpan file ini.")

        if "date_start" in sessions.columns:
            sessions["date_start"] = ensure_datetime_utc(sessions["date_start"], "date_start")
        else:
            raise SystemExit("Kolom 'date_start' tidak ada pada sessions_race_upto_austin.csv.")

        ses_cols = ["session_key", "date_start"]
        for c in ["location", "country_name", "meeting_name", "session_name"]:
            if c in sessions.columns:
                ses_cols.append(c)
        results = results.merge(
            sessions[ses_cols].drop_duplicates("session_key"),
            on="session_key", how="left"
        )

        results["date_start"] = ensure_datetime_utc(results["date_start"], "date_start")

        base = results.copy()

        if "driver_name" not in base.columns and "broadcast_name" in base.columns:
            base["driver_name"] = base["broadcast_name"]

        keep_cols = [c for c in [
            "session_key", "driver_number", "driver_name", "broadcast_name", "team_name",
            "position", "points", "status", "date_start"
        ] if c in base.columns]

        base = base[keep_cols].copy()

        if "driver_number" not in base.columns and not drivers.empty and "driver_number" in drivers.columns:
            dn = drivers[["session_key","driver_number","broadcast_name"]].drop_duplicates()
            base = base.merge(dn, on=["session_key","broadcast_name"], how="left")

        sort_cols = [c for c in ["driver_number", "date_start"] if c in base.columns]
        if not sort_cols:
            sort_cols = ["date_start"]
        base = base.sort_values(sort_cols)

        def rolling3_mean(s): return s.shift().rolling(3, min_periods=1).mean()
        def rolling3_min(s):  return s.shift().rolling(3, min_periods=1).min()
        def rolling3_apply(s, func): return s.shift().rolling(3, min_periods=1).apply(func, raw=False)

        if "position" in base.columns:
            base["position_num"] = pd.to_numeric(base["position"], errors="coerce")
        else:
            base["position_num"] = math.nan

        if "status" in base.columns:
            base["dnf_flag"] = base["status"].astype(str).str.lower().apply(lambda x: 0 if "finish" in x else 1)
        else:
            base["dnf_flag"] = 0

        base["avg_finish_last3"] = base.groupby("driver_number")["position_num"].transform(rolling3_mean)
        base["best_finish_last3"] = base.groupby("driver_number")["position_num"].transform(rolling3_min)
        base["dnf_rate_last3"] = base.groupby("driver_number")["dnf_flag"].transform(
            lambda s: s.shift().rolling(3, min_periods=1).mean()
        )
        if "points" in base.columns:
            base["points_last3"] = base.groupby("driver_number")["points"].transform(
                lambda s: s.shift().rolling(3, min_periods=1).sum()
            )
        else:
            base["points_last3"] = math.nan

        if not grid.empty and {"session_key","driver_number","position"}.issubset(grid.columns):
            g = grid.rename(columns={"position":"grid_pos"})[["session_key","driver_number","grid_pos"]]
            g["grid_pos"] = pd.to_numeric(g["grid_pos"], errors="coerce")
            base = base.merge(g, on=["session_key","driver_number"], how="left")
            base["avg_grid_pos_last3"] = base.groupby("driver_number")["grid_pos"].transform(rolling3_mean)
        else:
            base["grid_pos"] = math.nan
            base["avg_grid_pos_last3"] = math.nan

        if not weather.empty and {"session_key","air_temperature","track_temperature"}.issubset(weather.columns):
            w = weather.groupby("session_key")[["air_temperature","track_temperature"]].mean().reset_index()
            base = base.merge(w, on="session_key", how="left")
        else:
            base["air_temperature"] = math.nan
            base["track_temperature"] = math.nan

        if not laps.empty and {"session_key","driver_number","lap_number","lap_duration"}.issubset(laps.columns):
            if "lap_time" in laps.columns and "lap_duration" not in laps.columns:
                laps["lap_duration"] = laps["lap_time"]
            laps["lap_duration"] = pd.to_timedelta(laps["lap_duration"], errors="coerce")
            pace = laps.groupby(["session_key","driver_number"])["lap_duration"].agg(
                median_lap_time=lambda s: s.dropna().median(),
                best_lap_time=lambda s: s.dropna().min()
            ).reset_index()

            for col in ["median_lap_time","best_lap_time"]:
                pace[col] = pace[col].dt.total_seconds()
            base = base.merge(pace, on=["session_key","driver_number"], how="left")

            for col in ["median_lap_time","best_lap_time"]:
                base[f"{col}_last3"] = base.groupby("driver_number")[col].transform(
                    lambda s: s.shift().rolling(3, min_periods=1).mean()
                )
        else:
            base["median_lap_time"] = math.nan
            base["best_lap_time"] = math.nan
            base["median_lap_time_last3"] = math.nan
            base["best_lap_time_last3"] = math.nan

        if "team_name" in base.columns and "points" in base.columns:
            base["team_avg_points_last3"] = base.groupby("team_name")["points"].transform(
                lambda s: s.shift().rolling(3, min_periods=1).mean()
            )
            base["team_best_finish_last3"] = base.groupby("team_name")["position_num"].transform(rolling3_min)
        else:
            base["team_avg_points_last3"] = math.nan
            base["team_best_finish_last3"] = math.nan

        base_cut = base[base["date_start"] <= self.cutoff].copy()

        base_cut["track_altitude_m"] = self.cfg.track_altitude_m
        base_cut["track_length_km"] = self.cfg.track_length_km
        base_cut["overtake_difficulty"] = self.cfg.overtake_difficulty
        base_cut["track_type"] = self.cfg.track_type

        final_cols = [c for c in [
            "session_key", "date_start",
            "driver_number","driver_name","broadcast_name","team_name",
            "position_num","points","dnf_flag","grid_pos",
            "avg_finish_last3","best_finish_last3","dnf_rate_last3","points_last3",
            "avg_grid_pos_last3",
            "median_lap_time","best_lap_time","median_lap_time_last3","best_lap_time_last3",
            "team_avg_points_last3","team_best_finish_last3",
            "air_temperature","track_temperature",
            "track_altitude_m","track_length_km","overtake_difficulty","track_type"
        ] if c in base_cut.columns]

        feats = base_cut[final_cols].copy()
        feats = feats.sort_values(["date_start","driver_number"])
        feats.to_csv(self.cfg.out_features_csv, index=False)
        print(f"Dah kesimpen: {self.cfg.out_features_csv}")

        return feats

def main():
    ap = argparse.ArgumentParser(description="OpenF1 2025 → Features for Mexico GP winner prediction")
    ap.add_argument("--outdir", default="openf1_2025_until_austin", help="Folder output data mentah")
    ap.add_argument("--no-heavy", action="store_true", help="Skip download laps/pit/stints/weather/car_data")
    ap.add_argument("--car-speed-filter", type=int, default=300, help="Filter kecepatan minimum untuk car_data")
    ap.add_argument("--features-csv", default="features_up_to_austin_for_mexico.csv", help="Nama file fitur output (CSV)")
    ap.add_argument("--track-altitude", type=float, default=2285.0)
    ap.add_argument("--track-length", type=float, default=4.304)
    ap.add_argument("--overtake-diff", type=float, default=0.65)
    ap.add_argument("--track-type", default="high-altitude, medium-downforce")

    args = ap.parse_args()

    fetch_cfg = FetchConfig(
        outdir=args.outdir,
        download_heavy=(not args.no_heavy),
        car_speed_filter=args.car_speed_filter
    )
    fetcher = OpenF1Fetcher(fetch_cfg)
    fetch_result = fetcher.fetch_all_until_austin()

    feat_cfg = FeatureConfig(
        out_features_csv=args.features_csv,
        track_altitude_m=args.track_altitude,
        track_length_km=args.track_length,
        overtake_difficulty=args.overtake_diff,
        track_type=args.track_type
    )
    fb = FeatureBuilder(args.outdir, feat_cfg, cutoff_utc=fetch_result["cutoff_utc"])
    fb.build()

if __name__ == "__main__":
    main()
