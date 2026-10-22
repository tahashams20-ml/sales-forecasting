"""Flask application serving the Walmart Sales Forecasting dashboard + JSON API."""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    "Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "Year", "Quarter", "Month", "DayOfYear", "DayOfWeek",
    "Sales_Lag_1", "Sales_Lag_4", "Sales_Lag_52", "Rolling_Mean_4", "Rolling_Mean_12",
]
LAG_COLS = ["Sales_Lag_1", "Sales_Lag_4", "Sales_Lag_52", "Rolling_Mean_4", "Rolling_Mean_12"]
MAX_HORIZON = 52
Z80 = 1.2816  # z-score for an ~80% two-sided interval


def _holiday_name(date: pd.Timestamp) -> str:
    """Map a Walmart holiday week (Holiday_Flag == 1) to its event name."""
    m = date.month
    if m == 2:
        return "Super Bowl"
    if m == 9:
        return "Labor Day"
    if m == 11:
        return "Thanksgiving"
    if m == 12:
        return "Christmas"
    return "Holiday"


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.quarter
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df


@dataclass
class Scenario:
    temperature_delta: float = 0.0   # °F added to seasonal temperature
    fuel_price_pct: float = 0.0      # % change applied to last known fuel price
    cpi_pct: float = 0.0             # % change applied to last known CPI
    unemployment_delta: float = 0.0  # percentage points added to last known unemployment
    holiday_mode: str = "auto"       # auto | none | all

    @property
    def is_baseline(self) -> bool:
        return (self.temperature_delta == 0 and self.fuel_price_pct == 0 and self.cpi_pct == 0
                and self.unemployment_delta == 0 and self.holiday_mode == "auto")


@dataclass
class ForecastEngine:
    base_dir: str
    model: object = field(init=False)
    raw: pd.DataFrame = field(init=False)
    featured: pd.DataFrame = field(init=False)
    test: pd.DataFrame = field(init=False)
    importance: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.model = joblib.load(os.path.join(self.base_dir, "walmart_sales_model.joblib"))
        raw = pd.read_csv(os.path.join(self.base_dir, "Walmart.csv"))
        raw["Date"] = pd.to_datetime(raw["Date"], format="%d-%m-%Y", errors="coerce")
        raw = raw.dropna(subset=["Date", "Weekly_Sales", "Store"])
        raw["Store"] = raw["Store"].astype(int)
        self.raw = raw.sort_values(["Store", "Date"]).reset_index(drop=True)
        self.importance = pd.read_csv(os.path.join(self.base_dir, "walmart_feature_importance.csv"))

        self.stores = sorted(self.raw["Store"].unique().tolist())
        self.last_date = self.raw["Date"].max()
        self.first_date = self.raw["Date"].min()

        self._build_features()
        self._evaluate()
        self._extract_coefficients()
        self.baseline = self.forecast(MAX_HORIZON, Scenario())

    def _build_features(self):
        df = _add_date_features(self.raw.copy())
        for lag in [1, 4, 52]:
            df[f"Sales_Lag_{lag}"] = df.groupby("Store")["Weekly_Sales"].shift(lag)
        for w in [4, 12]:
            df[f"Rolling_Mean_{w}"] = df.groupby("Store")["Weekly_Sales"].transform(
                lambda x, w=w: x.shift(1).rolling(window=w).mean())
        df = df.dropna(subset=LAG_COLS).sort_values(["Date", "Store"]).reset_index(drop=True)
        self.featured = df

        unique_dates = np.sort(df["Date"].unique())
        split_idx = int(len(unique_dates) * 0.8)
        self.cutoff_date = pd.Timestamp(unique_dates[split_idx])
        self.train_period = (df["Date"].min(), self.cutoff_date - pd.Timedelta(days=7))
        self.test = df[df["Date"] >= self.cutoff_date].copy()
        self.n_train_rows = int((df["Date"] < self.cutoff_date).sum())

    def _evaluate(self):
        t = self.test
        t["Predicted"] = self.model.predict(t[FEATURES])
        t["Error"] = t["Weekly_Sales"] - t["Predicted"]
        y, p = t["Weekly_Sales"], t["Predicted"]
        mape = float(np.mean(np.abs(t["Error"]) / y) * 100)

        def _m(pred):
            return {
                "mae": float(mean_absolute_error(y, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                "r2": float(r2_score(y, pred)),
                "mape": float(np.mean(np.abs(y - pred) / y) * 100),
            }

        self.metrics = {
            **_m(p),
            "mse": float(mean_squared_error(y, p)),
            "accuracy": 100 - mape,
            "test_rows": int(len(t)),
            "train_rows": self.n_train_rows,
            "test_start": t["Date"].min().strftime("%Y-%m-%d"),
            "test_end": t["Date"].max().strftime("%Y-%m-%d"),
            "train_start": self.train_period[0].strftime("%Y-%m-%d"),
            "train_end": self.train_period[1].strftime("%Y-%m-%d"),
        }
        self.baselines = {
            "Linear Regression (model)": _m(p),
            "Naive: last week (lag 1)": _m(t["Sales_Lag_1"]),
            "Seasonal naive (lag 52)": _m(t["Sales_Lag_52"]),
            "Rolling mean (4 weeks)": _m(t["Rolling_Mean_4"]),
        }
        g = t.groupby("Store")
        self.store_stats = pd.DataFrame({
            "mae": g["Error"].apply(lambda e: e.abs().mean()),
            "rmse": g["Error"].apply(lambda e: np.sqrt((e ** 2).mean())),
            "mape": g.apply(lambda d: (d["Error"].abs() / d["Weekly_Sales"]).mean() * 100,
                            include_groups=False),
        })

    def _extract_coefficients(self):
        pre = self.model.named_steps["preprocessor"]
        reg = self.model.named_steps["regressor"]
        names = list(pre.get_feature_names_out())
        self.coef = dict(zip(names, reg.coef_.tolist()))
        self.intercept = float(reg.intercept_)

    def forecast(self, horizon: int, scenario: Scenario, stores: list[int] | None = None) -> pd.DataFrame:
        horizon = int(max(1, min(horizon, MAX_HORIZON)))
        stores = stores or self.stores
        hist = self.raw[self.raw["Store"].isin(stores)]
        series = {s: g.set_index("Date") for s, g in hist.groupby("Store")}
        sales = {s: g["Weekly_Sales"].tolist() for s, g in series.items()}
        out = []
        for h in range(1, horizon + 1):
            d = self.last_date + pd.Timedelta(weeks=h)
            ref_date = d - pd.Timedelta(weeks=52)
            rows = []
            for s, g in series.items():
                last = g.iloc[-1]
                ref = g.loc[ref_date] if ref_date in g.index else last
                hf = int(ref["Holiday_Flag"])
                if scenario.holiday_mode == "none":
                    hf = 0
                elif scenario.holiday_mode == "all":
                    hf = 1
                sv = sales[s]
                rows.append({
                    "Store": s, "Date": d, "Holiday_Flag": hf,
                    "Temperature": float(ref["Temperature"]) + scenario.temperature_delta,
                    "Fuel_Price": float(last["Fuel_Price"]) * (1 + scenario.fuel_price_pct / 100),
                    "CPI": float(last["CPI"]) * (1 + scenario.cpi_pct / 100),
                    "Unemployment": float(last["Unemployment"]) + scenario.unemployment_delta,
                    "Sales_Lag_1": sv[-1], "Sales_Lag_4": sv[-4], "Sales_Lag_52": sv[-52],
                    "Rolling_Mean_4": float(np.mean(sv[-4:])),
                    "Rolling_Mean_12": float(np.mean(sv[-12:])),
                    "Last_Year_Sales": float(ref["Weekly_Sales"]) if ref_date in g.index else np.nan,
                })
            step = _add_date_features(pd.DataFrame(rows))
            step["Forecast"] = self.model.predict(step[FEATURES])
            for s, val in zip(step["Store"], step["Forecast"]):
                sales[s].append(float(val))
            step["Horizon"] = h
            out.append(step)
        return pd.concat(out, ignore_index=True)

    def get_forecast(self, horizon: int, stores: list[int] | None, scenario: Scenario | None = None):
        scenario = scenario or Scenario()
        if scenario.is_baseline:
            fc = self.baseline[self.baseline["Horizon"] <= horizon]
            if stores:
                fc = fc[fc["Store"].isin(stores)]
            return fc.copy()
        return self.forecast(horizon, scenario, stores)

    def interval_sigma(self, stores: list[int] | None) -> float:
        t = self.test if not stores else self.test[self.test["Store"].isin(stores)]
        agg = t.groupby("Date")["Error"].sum()
        return float(agg.std(ddof=1))

    def predict_single(self, store: int, date: pd.Timestamp, overrides: dict) -> dict:
        g = self.raw[self.raw["Store"] == store].set_index("Date")
        date = pd.Timestamp(date)
        offset = (date - self.first_date).days
        date = self.first_date + pd.Timedelta(weeks=round(offset / 7))
        if date < self.first_date + pd.Timedelta(weeks=52):
            raise ValueError("Date must be at least 52 weeks after the start of the dataset "
                             f"({(self.first_date + pd.Timedelta(weeks=52)).date()}) so lag-52 exists.")
        if date > self.last_date + pd.Timedelta(weeks=MAX_HORIZON):
            raise ValueError(f"Date must be on/before {(self.last_date + pd.Timedelta(weeks=MAX_HORIZON)).date()}.")

        actual = None
        mode = "historical"
        if date <= self.last_date:
            hist_sales = g.loc[:date - pd.Timedelta(weeks=1), "Weekly_Sales"].tolist()
            base = g.loc[date]
            actual = float(base["Weekly_Sales"])
            defaults = {k: float(base[k]) for k in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]}
            defaults["Holiday_Flag"] = int(base["Holiday_Flag"])
        else:
            mode = "future"
            h = int((date - self.last_date).days // 7)
            fc = self.baseline[(self.baseline["Store"] == store) & (self.baseline["Horizon"] <= h)]
            hist_sales = g["Weekly_Sales"].tolist() + fc[fc["Horizon"] < h]["Forecast"].tolist()
            row = fc[fc["Horizon"] == h].iloc[0]
            defaults = {k: float(row[k]) for k in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]}
            defaults["Holiday_Flag"] = int(row["Holiday_Flag"])

        feats = {**defaults}
        for k in ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"]:
            if overrides.get(k) is not None and overrides.get(k) != "":
                feats[k] = float(overrides[k]) if k != "Holiday_Flag" else int(overrides[k])
        feats.update({
            "Store": store,
            "Sales_Lag_1": hist_sales[-1], "Sales_Lag_4": hist_sales[-4], "Sales_Lag_52": hist_sales[-52],
            "Rolling_Mean_4": float(np.mean(hist_sales[-4:])),
            "Rolling_Mean_12": float(np.mean(hist_sales[-12:])),
        })
        X = _add_date_features(pd.DataFrame([{**feats, "Date": date}]))
        pred = float(self.model.predict(X[FEATURES])[0])

        contrib = [{"feature": "Intercept", "value": None, "contribution": self.intercept}]
        store_key = f"store__Store_{store}"
        contrib.append({"feature": f"Store {store} effect", "value": store,
                        "contribution": self.coef.get(store_key, 0.0)})
        for f in FEATURES[1:]:
            v = float(X.iloc[0][f])
            contrib.append({"feature": f, "value": v, "contribution": self.coef[f"remainder__{f}"] * v})

        st = self.store_stats.loc[store]
        return {
            "store": store, "date": date.strftime("%Y-%m-%d"), "mode": mode,
            "prediction": pred, "actual": actual,
            "interval": [pred - Z80 * float(st["rmse"]), pred + Z80 * float(st["rmse"])],
            "store_mape": float(st["mape"]),
            "features": {k: (float(X.iloc[0][k]) if k != "Store" else store) for k in FEATURES},
            "defaults": defaults,
            "contributions": contrib,
        }


app = Flask(__name__)
engine = ForecastEngine(BASE_DIR)



# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _stores_param(value: str | None) -> list[int] | None:
    if not value or value == "all":
        return None
    ids = [int(v) for v in str(value).split(",") if v.strip()]
    bad = [i for i in ids if i not in engine.stores]
    if bad:
        raise ValueError(f"Unknown store(s): {bad}")
    return ids


def _horizon_param(value) -> int:
    try:
        h = int(value)
    except (TypeError, ValueError):
        h = 12
    return max(1, min(h, MAX_HORIZON))


def _scenario_from(data: dict) -> Scenario:
    return Scenario(
        temperature_delta=float(data.get("temperature_delta", 0) or 0),
        fuel_price_pct=float(data.get("fuel_price_pct", 0) or 0),
        cpi_pct=float(data.get("cpi_pct", 0) or 0),
        unemployment_delta=float(data.get("unemployment_delta", 0) or 0),
        holiday_mode=data.get("holiday_mode", "auto") if data.get("holiday_mode") in ("auto", "none", "all") else "auto",
    )


def _ds(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _pct(a, b):
    return None if not b else float((a - b) / b * 100)


def _forecast_series(fc: pd.DataFrame, stores):
    sigma = engine.interval_sigma(stores)
    agg = fc.groupby(["Date", "Horizon"], as_index=False).agg(
        forecast=("Forecast", "sum"), last_year=("Last_Year_Sales", "sum"), holiday=("Holiday_Flag", "max"))
    out = []
    for r in agg.itertuples():
        band = Z80 * sigma * np.sqrt(r.Horizon)
        out.append({
            "date": _ds(r.Date), "horizon": int(r.Horizon), "forecast": float(r.forecast),
            "lo": float(r.forecast - band), "hi": float(r.forecast + band),
            "last_year": float(r.last_year), "holiday": int(r.holiday),
            "holiday_name": _holiday_name(r.Date) if r.holiday else None,
        })
    return out


@app.errorhandler(ValueError)
def _bad_request(e):
    return jsonify({"error": str(e)}), 400


# ---------------------------------------------------------------------- #
# pages
# ---------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": engine.model is not None}


# ---------------------------------------------------------------------- #
# API
# ---------------------------------------------------------------------- #
@app.route("/api/meta")
def meta():
    raw = engine.raw
    last = raw[raw["Date"] == engine.last_date].set_index("Store")
    return jsonify({
        "stores": engine.stores,
        "first_date": _ds(engine.first_date),
        "last_date": _ds(engine.last_date),
        "max_horizon": MAX_HORIZON,
        "n_rows": int(len(raw)),
        "n_weeks": int(raw["Date"].nunique()),
        "features": FEATURES,
        "model": type(engine.model.named_steps["regressor"]).__name__,
        "metrics": engine.metrics,
        "min_predict_date": _ds(engine.first_date + pd.Timedelta(weeks=52)),
        "max_predict_date": _ds(engine.last_date + pd.Timedelta(weeks=MAX_HORIZON)),
        "latest": {int(s): {k: float(last.loc[s, k]) for k in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]}
                   for s in last.index},
    })


@app.route("/api/dashboard")
def dashboard():
    stores = _stores_param(request.args.get("store"))
    horizon = _horizon_param(request.args.get("horizon", 12))
    raw = engine.raw if not stores else engine.raw[engine.raw["Store"].isin(stores)]
    test = engine.test if not stores else engine.test[engine.test["Store"].isin(stores)]

    hist = raw.groupby("Date").agg(actual=("Weekly_Sales", "sum"), holiday=("Holiday_Flag", "max")).reset_index()
    tp = test.groupby("Date").agg(actual=("Weekly_Sales", "sum"), predicted=("Predicted", "sum")).reset_index()
    fc_all = engine.get_forecast(horizon, stores)
    fseries = _forecast_series(fc_all, stores)

    total_fc = sum(p["forecast"] for p in fseries)
    total_ly = sum(p["last_year"] for p in fseries)
    last_actual = float(hist["actual"].iloc[-1])
    next_week = fseries[0]["forecast"]

    # Selection-level accuracy on the test set
    sel_mape = float((test["Error"].abs() / test["Weekly_Sales"]).mean() * 100)
    agg_err = tp["actual"] - tp["predicted"]
    sel_r2 = float(1 - (agg_err ** 2).sum() / ((tp["actual"] - tp["actual"].mean()) ** 2).sum())

    # Store ranking by forecast over horizon
    by_store = fc_all.groupby("Store").agg(forecast=("Forecast", "sum"), last_year=("Last_Year_Sales", "sum"))
    by_store["growth"] = (by_store["forecast"] - by_store["last_year"]) / by_store["last_year"] * 100
    by_store = by_store.join(engine.store_stats)
    rank_src = engine.get_forecast(horizon, None).groupby("Store").agg(
        forecast=("Forecast", "sum"), last_year=("Last_Year_Sales", "sum"))
    rank_src["growth"] = (rank_src["forecast"] - rank_src["last_year"]) / rank_src["last_year"] * 100
    top = rank_src.sort_values("forecast", ascending=False).head(10)

    # Holiday impact computed from the historical data of the selection
    h = raw.copy()
    h["Event"] = [(_holiday_name(d) if f else "Non-holiday") for d, f in zip(h["Date"], h["Holiday_Flag"])]
    ev = h.groupby("Event")["Weekly_Sales"].mean()
    non = float(ev.get("Non-holiday", np.nan))
    holiday_impact = [{"event": e, "avg_sales": float(v), "lift": _pct(v, non)}
                      for e, v in ev.items()]
    order = ["Non-holiday", "Super Bowl", "Labor Day", "Thanksgiving", "Christmas"]
    holiday_impact.sort(key=lambda x: order.index(x["event"]) if x["event"] in order else 99)

    # Monthly seasonality (avg weekly sales per calendar month)
    m = raw.assign(Month=raw["Date"].dt.month).groupby(["Month", "Store"])["Weekly_Sales"].mean() \
        .groupby("Month").sum()
    seasonality = [{"month": int(k), "avg_weekly": float(v)} for k, v in m.items()]

    # Data-driven alerts
    alerts = []
    peak = max(fseries, key=lambda p: p["forecast"])
    alerts.append({"type": "peak", "title": "Peak demand week",
                   "text": f"Highest forecast in the next {horizon} weeks is week ending {peak['date']}"
                           f"{' (' + peak['holiday_name'] + ')' if peak['holiday_name'] else ''}.",
                   "value": peak["forecast"]})
    for p in fseries:
        if p["holiday"]:
            alerts.append({"type": "holiday", "title": f"{p['holiday_name']} week ahead",
                           "text": f"Week ending {p['date']} is a holiday week. Forecast vs same week last year: "
                                   f"{_pct(p['forecast'], p['last_year']):+.1f}%.", "value": p["forecast"]})
    if not stores or len(stores) > 1:
        g = by_store.sort_values("growth")
        best, worst = g.iloc[-1], g.iloc[0]
        alerts.append({"type": "growth", "title": f"Store {int(g.index[-1])} outpacing last year",
                       "text": f"Forecast {best['growth']:+.1f}% vs the same {horizon} weeks last year.",
                       "value": float(best["forecast"])})
        alerts.append({"type": "decline", "title": f"Store {int(g.index[0])} trending below last year",
                       "text": f"Forecast {worst['growth']:+.1f}% vs the same {horizon} weeks last year. "
                               "Review staffing and inventory.", "value": float(worst["forecast"])})
        r = by_store.sort_values("mape").iloc[-1]
        alerts.append({"type": "risk", "title": f"Lower confidence: Store {int(by_store.sort_values('mape').index[-1])}",
                       "text": f"Model test MAPE is {r['mape']:.1f}% for this store. Keep a wider safety stock.",
                       "value": None})
    else:
        st = engine.store_stats.loc[stores[0]]
        alerts.append({"type": "risk", "title": "Store-level model reliability",
                       "text": f"Test MAPE {st['mape']:.2f}%, RMSE ${st['rmse']:,.0f} per week for Store {stores[0]}.",
                       "value": None})

    return jsonify({
        "selection": "All stores" if not stores else ", ".join(f"Store {s}" for s in stores),
        "horizon": horizon,
        "kpis": {
            "forecast_total": total_fc, "forecast_vs_ly": _pct(total_fc, total_ly), "last_year_total": total_ly,
            "next_week": next_week, "next_week_date": fseries[0]["date"], "next_week_vs_last": _pct(next_week, last_actual),
            "accuracy": 100 - sel_mape, "mape": sel_mape, "r2": sel_r2,
            "global_accuracy": engine.metrics["accuracy"],
            "top_store": int(top.index[0]), "top_store_forecast": float(top.iloc[0]["forecast"]),
            "top_store_growth": float(top.iloc[0]["growth"]),
            "peak_week": peak["date"], "peak_value": peak["forecast"],
        },
        "history": [{"date": _ds(r.Date), "actual": float(r.actual), "holiday": int(r.holiday)} for r in hist.itertuples()],
        "test_predictions": [{"date": _ds(r.Date), "actual": float(r.actual), "predicted": float(r.predicted)}
                             for r in tp.itertuples()],
        "forecast": fseries,
        "top_stores": [{"store": int(s), "forecast": float(r["forecast"]), "growth": float(r["growth"])}
                       for s, r in top.iterrows()],
        "holiday_impact": holiday_impact,
        "seasonality": seasonality,
        "alerts": alerts,
        "cutoff_date": _ds(engine.cutoff_date),
    })


@app.route("/api/stores")
def stores_table():
    horizon = _horizon_param(request.args.get("horizon", 12))
    raw = engine.raw
    fc = engine.get_forecast(horizon, None).groupby("Store").agg(
        forecast=("Forecast", "sum"), last_year=("Last_Year_Sales", "sum"))
    last52_cut = engine.last_date - pd.Timedelta(weeks=52)
    prev52_cut = engine.last_date - pd.Timedelta(weeks=104)
    rows = []
    for s, g in raw.groupby("Store"):
        l52 = g[g["Date"] > last52_cut]["Weekly_Sales"].sum()
        p52 = g[(g["Date"] > prev52_cut) & (g["Date"] <= last52_cut)]["Weekly_Sales"].sum()
        hol = g[g["Holiday_Flag"] == 1]["Weekly_Sales"].mean()
        non = g[g["Holiday_Flag"] == 0]["Weekly_Sales"].mean()
        st = engine.store_stats.loc[s]
        last = g.iloc[-1]
        rows.append({
            "store": int(s), "total_sales": float(g["Weekly_Sales"].sum()),
            "avg_weekly": float(g["Weekly_Sales"].mean()), "last_week": float(last["Weekly_Sales"]),
            "yoy_growth": _pct(l52, p52), "holiday_lift": _pct(hol, non),
            "forecast": float(fc.loc[s, "forecast"]), "forecast_growth": _pct(fc.loc[s, "forecast"], fc.loc[s, "last_year"]),
            "mape": float(st["mape"]), "mae": float(st["mae"]),
            "temperature": float(last["Temperature"]), "fuel_price": float(last["Fuel_Price"]),
            "cpi": float(last["CPI"]), "unemployment": float(last["Unemployment"]),
            "spark": [float(v) for v in g["Weekly_Sales"].tail(26)],
        })
    return jsonify({"horizon": horizon, "stores": rows})


@app.route("/api/performance")
def performance():
    t = engine.test
    counts, edges = np.histogram(t["Error"], bins=30)
    weekly = t.groupby("Date").apply(
        lambda d: pd.Series({"mape": (d["Error"].abs() / d["Weekly_Sales"]).mean() * 100,
                             "actual": d["Weekly_Sales"].sum(), "predicted": d["Predicted"].sum()}),
        include_groups=False).reset_index()
    imp = engine.importance.sort_values("Importance", ascending=False)
    coefs = [{"feature": f, "coef": engine.coef[f"remainder__{f}"]} for f in FEATURES[1:]]
    return jsonify({
        "metrics": engine.metrics,
        "baselines": [{"name": k, **v} for k, v in engine.baselines.items()],
        "importance": [{"feature": r.Feature, "importance": float(r.Importance), "std": float(r.Std)}
                       for r in imp.itertuples()],
        "error_hist": {"counts": counts.tolist(), "edges": edges.tolist()},
        "scatter": [{"x": float(a), "y": float(p), "store": int(s)}
                    for a, p, s in zip(t["Weekly_Sales"], t["Predicted"], t["Store"])],
        "weekly": [{"date": _ds(r.Date), "mape": float(r.mape), "actual": float(r.actual), "predicted": float(r.predicted)}
                   for r in weekly.itertuples()],
        "store_mape": [{"store": int(s), "mape": float(r["mape"]), "rmse": float(r["rmse"])}
                       for s, r in engine.store_stats.sort_values("mape").iterrows()],
        "coefficients": coefs,
        "intercept": engine.intercept,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    store = int(data.get("store", 1))
    if store not in engine.stores:
        raise ValueError(f"Unknown store {store}")
    if not data.get("date"):
        raise ValueError("A date is required.")
    return jsonify(engine.predict_single(store, pd.Timestamp(data["date"]), data))


@app.route("/api/scenario", methods=["POST"])
def scenario():
    data = request.get_json(force=True) or {}
    stores = _stores_param(data.get("store"))
    horizon = _horizon_param(data.get("horizon", 12))
    sc = _scenario_from(data)
    base = engine.get_forecast(horizon, stores)
    alt = engine.forecast(horizon, sc, stores)
    b = base.groupby("Date")["Forecast"].sum()
    a = alt.groupby("Date")["Forecast"].sum()
    ly = base.groupby("Date")["Last_Year_Sales"].sum()
    bs = base.groupby("Store")["Forecast"].sum()
    as_ = alt.groupby("Store")["Forecast"].sum()
    store_delta = ((as_ - bs) / bs * 100).sort_values()
    return jsonify({
        "horizon": horizon,
        "series": [{"date": _ds(d), "baseline": float(b[d]), "scenario": float(a[d]), "last_year": float(ly[d])}
                   for d in b.index],
        "baseline_total": float(b.sum()), "scenario_total": float(a.sum()),
        "delta": float(a.sum() - b.sum()), "delta_pct": _pct(a.sum(), b.sum()),
        "store_impact": [{"store": int(s), "delta_pct": float(v), "delta": float(as_[s] - bs[s])}
                         for s, v in store_delta.items()],
        "assumptions": {
            "temperature_delta": sc.temperature_delta, "fuel_price_pct": sc.fuel_price_pct,
            "cpi_pct": sc.cpi_pct, "unemployment_delta": sc.unemployment_delta, "holiday_mode": sc.holiday_mode,
        },
    })


@app.route("/api/export/forecast.csv")
def export_forecast():
    stores = _stores_param(request.args.get("store"))
    horizon = _horizon_param(request.args.get("horizon", 12))
    fc = engine.get_forecast(horizon, stores)
    cols = ["Date", "Store", "Horizon", "Forecast", "Last_Year_Sales", "Holiday_Flag",
            "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    out = fc[cols].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out = out.round(2).sort_values(["Date", "Store"])
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    name = f"walmart_forecast_{'all' if not stores else '-'.join(map(str, stores))}_{horizon}w.csv"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        from waitress import serve
        print(f"Serving Walmart Sales Forecasting on http://localhost:{port}")
        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")



