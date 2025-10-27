import os
import math
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple

def ensure_datetime_utc(s: pd.Series):
    return pd.to_datetime(s, utc=True, errors="coerce")

def try_import_xgb():
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"File tidak ditemukan: {path}")
    return pd.read_csv(path)

def get_session_order(sessions: pd.DataFrame) -> List[int]:
    if "date_start" in sessions.columns:
        sessions["date_start"] = ensure_datetime_utc(sessions["date_start"])
    ss = sessions.dropna(subset=["session_key"])
    if "date_start" in ss.columns:
        ss = ss.sort_values("date_start")
    return ss["session_key"].astype(int).tolist()

def to_int_series(s):
    try:
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    except Exception:
        return pd.Series([pd.NA] * len(s), dtype="Int64")

def compute_winners_robust(results: pd.DataFrame) -> pd.DataFrame:
    """Kembalikan dataframe winners: ['session_key','winner_driver_number'] robust terhadap variasi kolom."""
    df = results.copy()

    if "session_key" not in df.columns or "driver_number" not in df.columns:
        raise SystemExit("nggak punya kolom 'session_key' / 'driver_number'.")

    df["session_key"] = to_int_series(df["session_key"])
    df["driver_number"] = to_int_series(df["driver_number"])

    for c in ["position", "points", "duration"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["pos_num"] = pd.to_numeric(df["position"], errors="coerce")
    df["points_num"] = pd.to_numeric(df["points"], errors="coerce")
    df["duration_num"] = pd.to_numeric(df["duration"], errors="coerce")

    for c in ["dnf", "dns", "dsq"]:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].fillna(False).astype(bool)

    df["eligible"] = (~df["dnf"]) & (~df["dns"]) & (~df["dsq"])

    winners = []
    for sk, g in df.groupby("session_key"):
        if pd.isna(sk):
            continue
        gg = g.copy()

        cand = gg[gg["eligible"]].copy()
        if cand.empty:
            cand = gg.copy()

        # prioritas 1: posisi numerik terkecil
        # prioritas 2: points terbanyak
        # prioritas 3: duration terkecil
        cand = cand.sort_values(
            ["pos_num", "points_num", "duration_num"],
            ascending=[True, False, True],
            na_position="last"
        )
        row = cand.iloc[0]
        if pd.isna(row["driver_number"]):
            continue
        winners.append({"session_key": int(sk), "winner_driver_number": int(row["driver_number"])})

    win_df = pd.DataFrame(winners).drop_duplicates("session_key")
    return win_df

def build_train_table(
    feats: pd.DataFrame,
    results: pd.DataFrame,
    sessions: pd.DataFrame,
    min_history_races: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "date_start" in feats.columns:
        feats["date_start"] = ensure_datetime_utc(feats["date_start"])
    if "date_start" in results.columns:
        results["date_start"] = ensure_datetime_utc(results["date_start"])
    if "date_start" in sessions.columns:
        sessions["date_start"] = ensure_datetime_utc(sessions["date_start"])

    if "session_key" not in sessions.columns:
        raise SystemExit("❌ sessions_race_upto_austin.csv harus punya kolom 'session_key'.")
    order = get_session_order(sessions)
    if len(order) < 2:
        raise SystemExit("❌ Butuh ≥2 race untuk membuat pasangan (t -> t+1).")

    winners = compute_winners_robust(results)
    if winners.empty:
        raise SystemExit("❌ Tidak menemukan pemenang satupun di results. Cek isi CSV results_upto_austin.csv")

    if "session_key" not in feats.columns or "driver_number" not in feats.columns:
        raise SystemExit("❌ fitur harus punya 'session_key' dan 'driver_number'.")
    feats["session_key"] = to_int_series(feats["session_key"])
    feats["driver_number"] = to_int_series(feats["driver_number"])

    rows = []

    for i in range(min_history_races, len(order) - 1):
        sk_t = int(order[i])       
        sk_next = int(order[i + 1]) 

        X_t = feats.loc[feats["session_key"].astype(int).eq(sk_t)].copy()
        if X_t.empty:
            continue

        win_row = winners.loc[winners["session_key"].eq(sk_next)]
        if win_row.empty:
            continue
        win_driver = int(win_row.iloc[0]["winner_driver_number"])

        X_t["won_next_race"] = (X_t["driver_number"].astype(int) == win_driver).astype(int)
        X_t["next_session_key"] = sk_next
        rows.append(X_t)

    if not rows:
        print(f"Debug: jumlah sessions terurut = {len(order)}, winners = {len(winners)}")
        print("Contoh winners head:\n", winners.head(5))
        raise SystemExit("❌ Tidak ada baris training yang terbentuk (mungkin winners belum tersedia untuk beberapa race t+1).")

    train_tbl = pd.concat(rows, ignore_index=True)

    sk_last = int(order[-1])
    X_mexico = feats.loc[feats["session_key"].astype(int).eq(sk_last)].copy()
    if X_mexico.empty:
        raise SystemExit("❌ Tidak menemukan fitur untuk Austin (race terakhir).")

    return train_tbl, X_mexico

def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_like = {
        "driver_name", "broadcast_name", "team_name", "status",
        "track_type", "date_start", "next_session_key",
        "session_key", "won_next_race"
    }

    cols = []
    for c in df.columns:
        if c in drop_like:
            continue
        if df[c].dtype.kind in "biufc":
            cols.append(c)
    return cols

def group_top1_accuracy(df_pred: pd.DataFrame) -> float:

    if df_pred.empty:
        return float("nan")
    accs = []
    for sk, g in df_pred.groupby("next_session_key"):
        if g["won_next_race"].max() != 1:
            continue
        top_driver = g.sort_values("pred_proba", ascending=False).iloc[0]
        accs.append(int(top_driver["won_next_race"] == 1))
    return float(np.mean(accs)) if accs else float("nan")

def attach_identities(X_df: pd.DataFrame, outdir: str, session_key_last: int) -> pd.DataFrame:
    import pandas as pd
    import os

    out = X_df.copy()

    path_drivers = os.path.join(outdir, "drivers_upto_austin.csv")
    if os.path.exists(path_drivers):
        drv = pd.read_csv(path_drivers)

        if "session_key" in drv.columns:
            drv["session_key"] = pd.to_numeric(drv["session_key"], errors="coerce").astype("Int64")
        if "driver_number" in drv.columns:
            drv["driver_number"] = pd.to_numeric(drv["driver_number"], errors="coerce").astype("Int64")

        name_candidates = [c for c in ["full_name","driver_name","name","broadcast_name","name_acronym","abbreviation"] if c in drv.columns]
        if name_candidates:
            drv["__driver_name_tmp"] = None
            for c in name_candidates:
                drv["__driver_name_tmp"] = drv["__driver_name_tmp"].fillna(drv[c])
        else:
            drv["__driver_name_tmp"] = None

        if "team_name" not in drv.columns:
            drv["team_name"] = None

        drv_sorted = drv.sort_values(["driver_number","session_key"], ascending=[True, True])
        latest_drv = drv_sorted.dropna(subset=["driver_number"]).groupby("driver_number", as_index=False).last()

        id_from_drivers = latest_drv[["driver_number","__driver_name_tmp","team_name"]].rename(
            columns={"__driver_name_tmp":"driver_name"}
        )

        if "driver_number" in out.columns:
            out["driver_number"] = pd.to_numeric(out["driver_number"], errors="coerce").astype("Int64")
            out = out.merge(id_from_drivers, on="driver_number", how="left", suffixes=("",""))

    path_results = os.path.join(outdir, "results_upto_austin.csv")
    if os.path.exists(path_results):
        res = pd.read_csv(path_results)

        if "session_key" in res.columns:
            res["session_key"] = pd.to_numeric(res["session_key"], errors="coerce").astype("Int64")
        if "driver_number" in res.columns:
            res["driver_number"] = pd.to_numeric(res["driver_number"], errors="coerce").astype("Int64")

        res_last = res[res["session_key"].astype("Int64") == pd.Series([session_key_last], dtype="Int64")[0]].copy()

        res_last["__driver_name_fallback"] = None
        for c in ["full_name","driver_name","name","broadcast_name","name_acronym","abbreviation"]:
            if c in res_last.columns:
                res_last["__driver_name_fallback"] = res_last["__driver_name_fallback"].fillna(res_last[c])

        keep_cols = ["driver_number"]
        if "__driver_name_fallback" in res_last.columns:
            keep_cols.append("__driver_name_fallback")
        if "team_name" in res_last.columns:
            keep_cols.append("team_name")

        if "driver_number" in out.columns and len(keep_cols) > 1:
            out = out.merge(res_last[keep_cols].drop_duplicates("driver_number"),
                            on="driver_number", how="left", suffixes=("",""))

            if "driver_name" in out.columns and "__driver_name_fallback" in out.columns:
                out["driver_name"] = out["driver_name"].fillna(out["__driver_name_fallback"])
            elif "__driver_name_fallback" in out.columns:
                out["driver_name"] = out["__driver_name_fallback"]

            if "team_name_x" in out.columns and "team_name_y" in out.columns:
                out["team_name"] = out["team_name_x"].fillna(out["team_name_y"])
                out = out.drop(columns=["team_name_x","team_name_y"])

            if "__driver_name_fallback" in out.columns:
                out = out.drop(columns=["__driver_name_fallback"])

    if "driver_name" not in out.columns:
        out["driver_name"] = None
    if "team_name" not in out.columns:
        out["team_name"] = None

    return out

def main():
    ap = argparse.ArgumentParser(description="Train model pemenang F1 dan prediksi GP Meksiko")
    ap.add_argument("--outdir", default="openf1_2025_until_austin", help="Folder data mentah")
    ap.add_argument("--features", default="features_up_to_austin_for_mexico.csv", help="CSV fitur dari pipeline")
    ap.add_argument("--pred_csv", default="prediction_mexico_probs.csv", help="Output ranking prediksi Mexico")
    ap.add_argument("--valid_last_k", type=int, default=3, help="Jumlah race terakhir untuk validasi (time-based)")
    ap.add_argument("--min_history_races", type=int, default=3, help="Minimal histori agar rolling features ada")
    args = ap.parse_args()

    feats = load_csv(args.features)
    results = load_csv(os.path.join(args.outdir, "results_upto_austin.csv"))
    sessions = load_csv(os.path.join(args.outdir, "sessions_race_upto_austin.csv"))

    train_tbl, X_mexico = build_train_table(
        feats=feats, results=results, sessions=sessions,
        min_history_races=args.min_history_races
    )

    feat_cols = pick_feature_columns(train_tbl)
    if not feat_cols:
        raise SystemExit("Nggak ada fitur numerik yang bisa dipakai untuk training.")

    unique_next = sorted(train_tbl["next_session_key"].unique())
    k = min(args.valid_last_k, len(unique_next) // 3 if len(unique_next) >= 6 else 1)
    valid_keys = set(unique_next[-k:]) if k > 0 else set([unique_next[-1]])

    train_mask = ~train_tbl["next_session_key"].isin(valid_keys)
    valid_mask = train_tbl["next_session_key"].isin(valid_keys)

    trX = train_tbl.loc[train_mask, feat_cols].fillna(0.0).values
    trY = train_tbl.loc[train_mask, "won_next_race"].values

    vaX = train_tbl.loc[valid_mask, feat_cols].fillna(0.0).values
    vaY = train_tbl.loc[valid_mask, "won_next_race"].values

    xgb = try_import_xgb()
    if xgb is not None:

        pos = max(1, trY.sum())
        neg = max(1, len(trY) - trY.sum())
        spw = float(neg) / float(pos)

        clf = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=3.0,
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
            scale_pos_weight=spw,
            eval_metric="logloss"
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=1000, max_depth=8, min_samples_leaf=2,
            class_weight={0:1.0, 1:50.0}, random_state=42
        )

    clf.fit(trX, trY)

    from sklearn.metrics import log_loss, roc_auc_score
    if len(np.unique(vaY)) > 1:
        va_pred = clf.predict_proba(vaX)[:, 1]
        try:
            ll = log_loss(vaY, va_pred, labels=[0,1])
        except Exception:
            ll = float("nan")
        try:
            auc = roc_auc_score(vaY, va_pred)
        except Exception:
            auc = float("nan")
        val_tbl = train_tbl.loc[valid_mask].copy()
        val_tbl["pred_proba"] = va_pred
        top1 = group_top1_accuracy(val_tbl)
        print(f"🔎 Valid — logloss: {ll:.4f}, AUC: {auc:.4f}, Top-1@race: {top1:.3f}")
    else:
        print("🔎 Valid — label tunggal, metrik AUC/logloss tidak relevan.")

    X_mex = X_mexico.copy()
    X_mex_feats = X_mex[feat_cols].fillna(0.0).values
    mex_probs = clf.predict_proba(X_mex_feats)[:, 1]

    X_mex["pred_win_proba"] = mex_probs

    order = sorted(sessions["session_key"].astype(int).tolist(), key=lambda sk: int(sk))
    sk_last = int(order[-1])
    X_mex = attach_identities(X_mex, outdir=args.outdir, session_key_last=sk_last)

    id_cols = [c for c in ["driver_number","driver_name","team_name"] if c in X_mex.columns]
    out_cols = id_cols + ["pred_win_proba"]
    out = X_mex[out_cols].sort_values("pred_win_proba", ascending=False).reset_index(drop=True)

    print("\n🏁 Top-10 GP Meksiko:")
    print(out.head(10).to_string(index=False, formatters={"pred_win_proba": lambda x: f"{x:.3f}"}))


if __name__ == "__main__":
    main()
