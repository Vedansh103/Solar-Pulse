# Simple EDA for the solar project (Phase 1: plots only, no model)
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

OUT_DIR = Path("output/eda_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for plant in [1, 2]:
    # Load the master file: one row per hour, time is the index
    df = pd.read_csv(f"output/master_hourly_plant{plant}.csv", index_col="timestamp", parse_dates=True)
    print(f"\n--- Plant {plant} ---")
    print(df[["ac_power_kw", "irradiation", "module_temperature", "ambient_temperature"]].describe().round(1))
    print("Missing generation hours:", df["ac_power_kw"].isna().sum())
    print("Correlation power vs irradiation:", round(df["ac_power_kw"].corr(df["irradiation"]), 3))

    # 1. Line plot: power and irradiation over time
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["ac_power_kw"], label="AC power (kW)")
    ax.set_ylabel("AC power (kW)")
    ax2 = ax.twinx()                                  # second y-axis for irradiation
    ax2.plot(df.index, df["irradiation"], color="orange", label="Irradiation")
    ax2.set_ylabel("Irradiation")
    ax.set_title(f"Plant {plant}: power and irradiation")
    plt.savefig(OUT_DIR / f"plant{plant}_master_1_line.png", bbox_inches="tight")

    # 2. Missing value plot: how many values are missing in each column
    plt.figure(figsize=(8, 4))
    df[["ac_power_kw", "irradiation", "module_temperature", "ambient_temperature"]].isna().sum().plot(kind="bar")
    plt.title(f"Plant {plant}: missing hours per column")
    plt.ylabel("Missing hours")
    plt.savefig(OUT_DIR / f"plant{plant}_master_2_missing.png", bbox_inches="tight")

    # 3. Box plot by hour of day: power at each hour
    df.boxplot(column="ac_power_kw", by="hour", figsize=(12, 4))
    plt.title(f"Plant {plant}: power by hour of day")
    plt.suptitle("")
    plt.xlabel("Hour of day")
    plt.ylabel("AC power (kW)")
    plt.savefig(OUT_DIR / f"plant{plant}_master_3_box.png", bbox_inches="tight")

    # 4. Histogram: how power values are spread
    plt.figure(figsize=(8, 4))
    df["ac_power_kw"].plot(kind="hist", bins=30)
    plt.title(f"Plant {plant}: histogram of power")
    plt.xlabel("AC power (kW)")
    plt.savefig(OUT_DIR / f"plant{plant}_master_4_hist.png", bbox_inches="tight")

    # 5. Seasonal plot: one line per day against hour of day
    plt.figure(figsize=(10, 5))
    for day, g in df.groupby(df.index.date):
        plt.plot(g["hour"], g["ac_power_kw"], alpha=0.5)
    plt.title(f"Plant {plant}: one line per day")
    plt.xlabel("Hour of day")
    plt.ylabel("AC power (kW)")
    plt.savefig(OUT_DIR / f"plant{plant}_master_5_seasonal.png", bbox_inches="tight")

    # 6. STL (optional): split into trend + daily pattern + residual
    # STL cannot handle gaps, so use the longest stretch with no missing values
    s = df["ac_power_kw"]
    block = (s.isna() != s.isna().shift()).cumsum()        # label each run of present/missing
    runs = s.groupby(block).agg(["size", lambda x: x.isna().any()])
    runs.columns = ["size", "has_nan"]
    best = runs[~runs["has_nan"]]["size"].idxmax()          # longest run without NaN
    s = s[block == best]
    result = STL(s, period=24).fit()                        # period 24 = one day
    result.plot()
    plt.savefig(OUT_DIR / f"plant{plant}_master_6_stl.png", bbox_inches="tight")

    # Peak-hour summary for the explanation
    print("Average power at 12:00:", round(df[df["hour"] == 12]["ac_power_kw"].mean(), 1), "kW")
    print("Average power at 00:00:", round(df[df["hour"] == 0]["ac_power_kw"].mean(), 1), "kW")
    print("Peak hour on average:", df.groupby("hour")["ac_power_kw"].mean().idxmax())
    print("STL trend start/end:", round(result.trend.iloc[0], 1), "->", round(result.trend.iloc[-1], 1))
    print("Residual std:", round(result.resid.std(), 1))

plt.close("all")
