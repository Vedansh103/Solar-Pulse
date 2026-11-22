"""Phase 1 preprocessing: one pipeline, run separately per plant (never stacked).

Inputs (in data/raw), outputs (in data/processed):
  master_hourly_plant{1,2}.csv, inverter_hourly_plant{1,2}.csv, quality_report.md
Phase 1 flags and never alters values: no filling, no row removal.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

START, END = "2020-05-15 00:00", "2020-06-17 23:00"
N_INV = 22
DAYLIGHT_THRESHOLD = 0.005

# Explicit timestamp formats (Plant 1 generation is day first)
GEN_FMT = {1: "%d-%m-%Y %H:%M", 2: "%Y-%m-%d %H:%M:%S"}
WX_FMT = "%Y-%m-%d %H:%M:%S"


def iqr_outliers(s):
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return int(((s < lo) | (s > hi)).sum()), lo, hi


def process(plant):
    gen = pd.read_csv(RAW / f"Plant_{plant}_Generation_Data.csv")
    wx = pd.read_csv(RAW / f"Plant_{plant}_Weather_Sensor_Data.csv")
    raw_gen_rows, raw_wx_rows = len(gen), len(wx)

    # Step 1: parse timestamps explicitly (all IST); drop columns per Section 4.2
    gen["ts"] = pd.to_datetime(gen["DATE_TIME"], format=GEN_FMT[plant])
    wx["ts"] = pd.to_datetime(wx["DATE_TIME"], format=WX_FMT)
    gen = gen[["ts", "SOURCE_KEY", "AC_POWER"]]
    wx = wx.drop(columns=["DATE_TIME", "PLANT_ID", "SOURCE_KEY"])
    dups = int(gen.duplicated(["ts", "SOURCE_KEY"]).sum() + wx["ts"].duplicated().sum())

    # Step 2: per 15-minute slot, mean AC over reporting inverters + count
    slot = gen.groupby("ts")["AC_POWER"].agg(ac="mean", n_inv="count")

    # Step 3: hourly mean of slots; keep mean inverter count, slots present, min count
    slot["hour_ts"] = slot.index.floor("h")
    hourly = slot.groupby("hour_ts").agg(
        ac_power_kw=("ac", "mean"),
        inverters_reporting=("n_inv", "mean"),
        slots_present=("ac", "count"),
        min_inv=("n_inv", "min"),
    )

    # Step 4: weather hourly means
    wx["hour_ts"] = wx["ts"].dt.floor("h")
    wxh = wx.groupby("hour_ts").agg(
        irradiation=("IRRADIATION", "mean"),
        module_temperature=("MODULE_TEMPERATURE", "mean"),
        ambient_temperature=("AMBIENT_TEMPERATURE", "mean"),
        wx_slots=("IRRADIATION", "count"),
    )

    # Step 5: continuous hourly index first, then left join generation + weather
    idx = pd.date_range(START, END, freq="h", name="timestamp")
    m = pd.DataFrame(index=idx).join(hourly).join(wxh)

    # Step 6: derived columns
    m["slots_present"] = m["slots_present"].fillna(0).astype(int)
    m["hour"] = m.index.hour
    m["has_generation"] = m["ac_power_kw"].notna().astype(int)
    m["has_weather"] = m["irradiation"].notna().astype(int)
    # Threshold above zero: the Plant 2 sensor shows noise up to about 0.002 at night
    m["is_daylight"] = (m["irradiation"] > DAYLIGHT_THRESHOLD).astype("Int64").where(m["has_weather"] == 1)
    m["flag_low_coverage"] = (m["min_inv"] < N_INV).astype("Int64").where(m["has_generation"] == 1)
    cols = ["ac_power_kw", "inverters_reporting", "slots_present", "irradiation",
            "module_temperature", "ambient_temperature", "hour", "is_daylight",
            "flag_low_coverage", "has_generation", "has_weather"]
    master = m[cols].copy()
    for c in ["ac_power_kw", "inverters_reporting", "irradiation", "module_temperature", "ambient_temperature"]:
        master[c] = master[c].round(4)

    # Step 7: assertions
    assert len(master) == 816
    assert master.index.is_unique
    assert pd.infer_freq(master.index) == "h"
    master.to_csv(OUT / f"master_hourly_plant{plant}.csv")

    # Inverter-level table (22 inverters x 816 hours) for Phase 3
    gen["hour_ts"] = gen["ts"].dt.floor("h")
    inv = gen.groupby(["hour_ts", "SOURCE_KEY"])["AC_POWER"].mean().unstack()
    inv = inv.reindex(idx).round(4)
    inv.index.name = "timestamp"
    inv.to_csv(OUT / f"inverter_hourly_plant{plant}.csv")

    # Quality stats
    n_all, lo_all, hi_all = iqr_outliers(master["ac_power_kw"])
    day = master.loc[master["is_daylight"] == 1, "ac_power_kw"]
    n_day, lo_day, hi_day = iqr_outliers(day)
    per_hour = 0
    for _, g in master.loc[master["is_daylight"] == 1].groupby("hour"):
        per_hour += iqr_outliers(g["ac_power_kw"])[0]
    inv_rows = gen.groupby("SOURCE_KEY").size()
    return dict(
        plant=plant, raw_gen=raw_gen_rows, raw_wx=raw_wx_rows, dups=dups,
        nulls=int(pd.read_csv(RAW / f"Plant_{plant}_Generation_Data.csv").isna().sum().sum()
                  + pd.read_csv(RAW / f"Plant_{plant}_Weather_Sensor_Data.csv").isna().sum().sum()),
        gen_slots=len(slot), wx_slots=int(wx["ts"].nunique()),
        no_gen=int((master["has_generation"] == 0).sum()),
        no_wx=int((master["has_weather"] == 0).sum()),
        low_cov=int((master["flag_low_coverage"] == 1).sum()),
        partial=int(((master["slots_present"] > 0) & (master["slots_present"] < 4)).sum()),
        out_all=(n_all, lo_all, hi_all), out_day=(n_day, lo_day, hi_day), out_hour=per_hour,
        n_day=len(day), night_zero=int(((master["is_daylight"] == 0) & (master["ac_power_kw"] == 0)).sum()),
        inv_min=int(inv_rows.min()), inv_max=int(inv_rows.max()),
        corr=master["ac_power_kw"].corr(master["irradiation"]),
        midday=master.loc[master["hour"].between(11, 13), "ac_power_kw"].median(),
    )


def report(stats):
    L = ["# Quality report (Phase 1)", "",
         "Phase 1 flags only: no values filled, altered or removed. Index is continuous, 816 hourly rows per plant (15 May 00:00 to 17 Jun 23:00 IST).", ""]
    hdr = "| Check | " + " | ".join(f"Plant {s['plant']}" for s in stats) + " |"
    L += [hdr, "|---|" + "---|" * len(stats)]
    rows = [
        ("Raw generation rows", "raw_gen"), ("Raw weather rows", "raw_wx"),
        ("Duplicate rows (gen + weather)", "dups"), ("Null values (raw files)", "nulls"),
        ("Distinct generation slots present (of 3,264)", "gen_slots"),
        ("Distinct weather slots present (of 3,264)", "wx_slots"),
        ("Hours with no generation data", "no_gen"), ("Hours with no weather data", "no_wx"),
        ("Hours with partial slots (1-3 of 4)", "partial"),
        ("Hours with some inverter not reporting (flag_low_coverage=1)", "low_cov"),
        ("Fewest / most rows per inverter", None),
        ("Hourly AC vs irradiation correlation", None),
        ("Median AC per inverter, 11:00-13:59 (kW)", None),
        ("Night zeros kept (daylight=0, AC=0)", "night_zero"),
        ("IQR outliers, all hours (fences)", None),
        ("IQR outliers, daylight hours only (fences)", None),
        ("IQR outliers, daylight, per hour of day", "out_hour"),
    ]
    for label, key in rows:
        vals = []
        for s in stats:
            if key:
                v = f"{s[key]:,}"
            elif label.startswith("Fewest"):
                v = f"{s['inv_min']:,} / {s['inv_max']:,}"
            elif label.startswith("Hourly AC"):
                v = f"{s['corr']:.3f}"
            elif label.startswith("Median"):
                v = f"{s['midday']:.0f}"
            elif "all hours" in label:
                v = f"{s['out_all'][0]} ({s['out_all'][1]:.0f}, {s['out_all'][2]:.0f})"
            else:
                v = f"{s['out_day'][0]} of {s['n_day']} ({s['out_day'][1]:.0f}, {s['out_day'][2]:.0f})"
            vals.append(v)
        L.append(f"| {label} | " + " | ".join(vals) + " |")
    L += ["", "## Notes", "",
          "- AC is the mean per reporting inverter; `inverters_reporting` is the hourly mean count, `flag_low_coverage` is 1 if any slot in the hour had fewer than 22 inverters (blank when no generation).",
          "- Outliers use the 1.5 x IQR rule and are flagged only. The all-hours rule is distorted by night zeros (bimodal data), so daylight-only and per-hour counts are also given.",
          "- DC_POWER, yields, PLANT_ID and sensor/raw-time columns are dropped from the master files; inverter SOURCE_KEY is kept only in `inverter_hourly_plantX.csv`.",
          "", "## Cleaning plan (Phase 2)", "",
          "- Interpolate short gaps (a few hours); flag long gaps (e.g. multi-hour overnight and 20 May daytime gap in Plant 1) rather than fill.",
          "- Treat Plant 2 low-coverage hours via the per-reporting-inverter mean already used here."]
    (OUT / "quality_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    stats = [process(p) for p in (1, 2)]
    report(stats)
    for s in stats:
        print(s)
