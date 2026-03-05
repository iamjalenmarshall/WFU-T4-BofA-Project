import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# ── 1. Load & reshape ──────────────────────────────────────────────────────────
df = pd.read_csv("Data.csv", header=0)
id_cols = df.columns[:2].tolist()
date_cols = df.columns[2:].tolist()

df_long = df.melt(id_vars=id_cols, value_vars=date_cols,
                  var_name="date", value_name="value")
df_long.columns = ["feature", "description", "date", "value"]
df_long["date"] = pd.to_datetime(df_long["date"])
df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

# Drop "Description not found" rows
df_long = df_long[df_long["description"] != "Description not found"]
df_long["label"] = df_long["description"].str.strip() + "\n(" + df_long["feature"] + ")"

features = sorted(df_long["label"].unique())
FORECAST_QUARTERS = 8   # 2 years ahead
CONFIDENCE = 0.95

# ── 2. Helper: choose differencing order via ADF test ─────────────────────────
def adf_d(series, max_d=2):
    for d in range(max_d + 1):
        s = series.diff(d).dropna() if d > 0 else series
        p = adfuller(s.dropna(), autolag="AIC")[1]
        if p < 0.05:
            return d
    return max_d

# ── 3. Fit ARIMA and forecast for one series ──────────────────────────────────
def fit_and_forecast(series):
    series = series.dropna()
    if len(series) < 12:
        return None, None, None, None

    d = adf_d(series)

    best_aic, best_order, best_model = np.inf, (1, d, 1), None
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                m = ARIMA(series, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic, best_order, best_model = m.aic, (p, d, q), m
            except Exception:
                continue

    if best_model is None:
        return None, None, None, None

    forecast = best_model.get_forecast(steps=FORECAST_QUARTERS)
    mean = forecast.predicted_mean
    ci   = forecast.conf_int(alpha=1 - CONFIDENCE)

    # Build future date index (quarterly)
    last_date = series.index[-1]
    future_idx = pd.date_range(start=last_date + pd.offsets.QuarterEnd(),
                               periods=FORECAST_QUARTERS, freq="QE")
    mean.index = future_idx
    ci.index   = future_idx

    return mean, ci, best_order, best_aic

# ── 4. Plot grid ───────────────────────────────────────────────────────────────
n      = len(features)
ncols  = 2
nrows  = (n + 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(18, nrows * 4), constrained_layout=True)
axes = axes.flatten()

results_summary = []

for i, feat in enumerate(features):
    ax = axes[i]
    subset = (df_long[df_long["label"] == feat]
              .sort_values("date")
              .set_index("date")["value"])

    # Scale to millions
    scale = 1e6
    series_m = subset / scale

    mean, ci, order, aic = fit_and_forecast(series_m)

    # Historical
    ax.plot(series_m.index, series_m.values, color="steelblue",
            linewidth=1.5, label="Historical")

    if mean is not None:
        # Forecast line
        ax.plot(mean.index, mean.values, color="darkorange",
                linewidth=1.8, linestyle="--", label=f"Forecast ARIMA{order}")
        # Confidence band
        ax.fill_between(ci.index,
                        ci.iloc[:, 0], ci.iloc[:, 1],
                        color="darkorange", alpha=0.2,
                        label=f"{int(CONFIDENCE*100)}% CI")
        # Connect last historical point to first forecast point
        ax.plot([series_m.index[-1], mean.index[0]],
                [series_m.values[-1], mean.values[0]],
                color="darkorange", linewidth=1.8, linestyle="--")

        results_summary.append({
            "Feature": feat.replace("\n", " "),
            "ARIMA Order": str(order),
            "AIC": round(aic, 1),
            f"Forecast Q1": round(mean.iloc[0], 1),
            f"Forecast Q8": round(mean.iloc[-1], 1),
        })
    else:
        results_summary.append({"Feature": feat.replace("\n", " "),
                                 "ARIMA Order": "insufficient data"})

    ax.set_title(feat, fontsize=7.5)
    ax.set_ylabel("$ Millions", fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=6, loc="upper left")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"ARIMA Forecasts — {FORECAST_QUARTERS} Quarter Horizon ({int(CONFIDENCE*100)}% CI)",
    fontsize=13, fontweight="bold")

plt.savefig("arima_forecasts.png", dpi=150, bbox_inches="tight")
print("✓ Chart saved: arima_forecasts.png")

# ── 5. Summary table ──────────────────────────────────────────────────────────
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv("arima_summary.csv", index=False)
print("✓ Summary saved: arima_summary.csv")
print("\n", summary_df.to_string(index=False))

plt.show()