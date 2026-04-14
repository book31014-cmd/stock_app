import os
import re
import uuid
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import shioaji as sj


# =========================
# Config
# =========================
SIMULATION = True

API_KEY = "API_KEY"
SECRET_KEY = "SECRET_KEY"

UNIVERSE = ["2330", "2317", "2603", "2881", "2890"]
MARKET_PROXY = "0050"

START_DATE = "2024-01-01"
END_DATE = "2026-04-13"

MODEL_PATH = "best_model.joblib"
TRADES_PATH = "trades.csv"
EQUITY_PATH = "equity_curve.csv"

INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.01
MAX_POSITIONS = 2
COOLDOWN_SECONDS = 180
MAX_DAILY_LOSS = -0.03

BUY_PROB_THRESHOLD = 0.60
BUY_NOT_TOO_EXTENDED = 0.02
BREAKOUT_LOOKBACK = 20
VOL_LOOKBACK = 20
ATR_LOOKBACK = 14
HOLDING_MINUTES_MAX = 30

FEATURE_COLS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ma5_ma20",
    "price_vwap",
    "volume_ratio",
    "atr_ratio",
    "breakout_20",
    "pullback_ma5",
    "trend_20",
    "intraday_pos",
    "volatility_20",
    "rsi14",
]

LABEL_HORIZON = 5
LABEL_THRESHOLD = 0.003


# =========================
# API / State
# =========================
api = sj.Shioaji(simulation=SIMULATION)
api.login(API_KEY, SECRET_KEY, subscribe_trade=True)

tick_buffer = defaultdict(list)     # stock -> list[dict]
kbar_live = {}                      # stock -> DataFrame
pending_orders = {}                 # custom_field -> order meta
positions = {}                      # stock -> Position
last_order_time = {}                # stock -> datetime
daily_pnl = 0.0
day_start_date = None
current_capital = INITIAL_CAPITAL
model = None
live_enabled = True


@dataclass
class Position:
    stock: str
    qty: int
    entry: float
    entry_time: datetime
    order_tag: str


# =========================
# Utils
# =========================
def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open("trade_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_contract(stock_code: str):
    return api.Contracts.Stocks[stock_code]


def safe_div(a, b):
    return a / b if b not in (0, None) and not pd.isna(b) else np.nan


def obj_to_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: obj_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [obj_to_dict(x) for x in obj]
    if hasattr(obj, "__dict__"):
        data = {}
        for k, v in obj.__dict__.items():
            if not k.startswith("_"):
                data[k] = obj_to_dict(v)
        return data
    return obj


def make_order_tag(stock: str):
    return f"{stock}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}-{uuid.uuid4().hex[:8]}"


# =========================
# Historical Data
# =========================
def normalize_kbars(raw):
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "ts": "ts",
    }
    df = df.rename(columns=rename_map)

    if "ts" not in df.columns:
        raise ValueError("kbars result has no ts column")

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    cols = [c for c in ["ts", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols]


def fetch_history_kbars(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    contract = get_contract(stock_code)
    raw = api.kbars(contract=contract, start=start_date, end=end_date)
    df = normalize_kbars(raw)
    if not df.empty:
        df["stock"] = stock_code
    return df


def fetch_universe_history(stocks, start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    for s in stocks:
        try:
            df = fetch_history_kbars(s, start_date, end_date)
            if not df.empty:
                frames.append(df)
                log(f"Fetched {s}: {len(df)} bars")
        except Exception as e:
            log(f"Fetch failed {s}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["stock", "ts"]).reset_index(drop=True)
    return out


# =========================
# Feature Engineering
# =========================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    df["date"] = df["ts"].dt.date
    g = df.groupby("date", group_keys=False)

    # session VWAP
    df["cum_vol"] = g["volume"].cumsum()
    df["cum_vp"] = g.apply(lambda x: (x["close"] * x["volume"]).cumsum())
    df["vwap"] = df["cum_vp"] / df["cum_vol"].replace(0, np.nan)

    # returns / trend
    df["ret_1"] = g["close"].pct_change(1)
    df["ret_3"] = g["close"].pct_change(3)
    df["ret_5"] = g["close"].pct_change(5)

    df["ma5"] = g["close"].transform(lambda x: x.rolling(5).mean())
    df["ma20"] = g["close"].transform(lambda x: x.rolling(20).mean())
    df["ma5_ma20"] = df["ma5"] / df["ma20"] - 1.0
    df["price_vwap"] = df["close"] / df["vwap"] - 1.0
    df["trend_20"] = df["ma5"] / df["ma20"] - 1.0

    # volume
    df["vol_mean20"] = g["volume"].transform(lambda x: x.rolling(VOL_LOOKBACK).mean())
    df["volume_ratio"] = df["volume"] / df["vol_mean20"]

    # breakout / pullback
    df["rolling_high20"] = g["high"].transform(lambda x: x.rolling(BREAKOUT_LOOKBACK).max())
    df["rolling_low20"] = g["low"].transform(lambda x: x.rolling(BREAKOUT_LOOKBACK).min())
    df["breakout_20"] = (df["close"] > df["rolling_high20"].shift(1)).astype(float)
    df["pullback_ma5"] = (
        (df["close"] > df["ma20"]) &
        (abs(df["close"] - df["ma5"]) / df["ma5"] < 0.01)
    ).astype(float)

    # intraday position
    rng = (df["rolling_high20"] - df["rolling_low20"]).replace(0, np.nan)
    df["intraday_pos"] = (df["close"] - df["rolling_low20"]) / rng

    # ATR
    prev_close = g["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["true_range"] = np.maximum.reduce([tr1.values, tr2.values, tr3.values])
    df["atr14"] = g["true_range"].transform(lambda x: x.rolling(ATR_LOOKBACK).mean())
    df["atr_ratio"] = df["atr14"] / df["close"]

    # volatility
    df["volatility_20"] = g["close"].transform(lambda x: x.pct_change().rolling(20).std())

    # RSI 14
    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = g["close"].transform(lambda x: x.diff().clip(lower=0).rolling(14).mean())
    avg_loss = g["close"].transform(lambda x: (-x.diff()).clip(lower=0).rolling(14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # labels
    df["future_ret"] = g["close"].shift(-LABEL_HORIZON) / df["close"] - 1.0
    df["label"] = (df["future_ret"] > LABEL_THRESHOLD).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    parts = []
    for stock, part in raw.groupby("stock"):
        feat = add_features(part.sort_values("ts"))
        feat["stock"] = stock
        parts.append(feat)

    data = pd.concat(parts, ignore_index=True)
    data = data.dropna(subset=FEATURE_COLS + ["label"]).sort_values(["ts", "stock"]).reset_index(drop=True)
    return data


# =========================
# Model
# =========================
def train_model(train_df: pd.DataFrame):
    X = train_df[FEATURE_COLS].fillna(0.0)
    y = train_df["label"].astype(int)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X, y)
    return clf


def save_model(clf):
    joblib.dump(clf, MODEL_PATH)
    log(f"Model saved: {MODEL_PATH}")


def load_model():
    if os.path.exists(MODEL_PATH):
        clf = joblib.load(MODEL_PATH)
        log(f"Model loaded: {MODEL_PATH}")
        return clf
    return None


# =========================
# Backtest
# =========================
@dataclass
class BacktestTrade:
    stock: str
    entry_ts: pd.Timestamp
    entry_price: float
    exit_ts: pd.Timestamp
    exit_price: float
    qty: int
    pnl_pct: float
    reason: str


def backtest_one_day(day_df: pd.DataFrame, clf, capital: float):
    positions_bt = {}
    trades = []
    cash_pnl = 0.0

    for stock, part in day_df.groupby("stock"):
        part = part.sort_values("ts").copy()
        if len(part) < 5:
            continue

        prob = clf.predict_proba(part[FEATURE_COLS].fillna(0.0))[:, 1]
        part["prob"] = prob

        for i in range(len(part) - 1):
            row = part.iloc[i]
            nxt = part.iloc[i + 1]

            # entry
            if stock not in positions_bt:
                if (
                    row["prob"] >= BUY_PROB_THRESHOLD and
                    row["ma5"] > row["ma20"] and
                    row["close"] > row["vwap"] and
                    row["volume_ratio"] >= 1.2 and
                    row["close"] <= row["ma5"] * (1 + BUY_NOT_TOO_EXTENDED)
                ):
                    entry = float(nxt["open"])
                    qty = max(int(capital * RISK_PER_TRADE / entry), 1)
                    atr = float(row["atr14"]) if pd.notna(row["atr14"]) else entry * 0.01
                    stop = entry - 1.5 * atr
                    take = entry + 2.0 * atr

                    positions_bt[stock] = {
                        "entry_ts": nxt["ts"],
                        "entry": entry,
                        "qty": qty,
                        "stop": stop,
                        "take": take,
                    }

            # exit
            if stock in positions_bt:
                pos = positions_bt[stock]
                exit_price = None
                reason = None

                if row["low"] <= pos["stop"]:
                    exit_price = pos["stop"]
                    reason = "stop"
                elif row["high"] >= pos["take"]:
                    exit_price = pos["take"]
                    reason = "take"
                elif i == len(part) - 2:
                    exit_price = float(nxt["close"])
                    reason = "eod"

                if exit_price is not None:
                    pnl_pct = (exit_price - pos["entry"]) / pos["entry"]
                    cash_pnl += pnl_pct * pos["qty"] * pos["entry"]
                    trades.append(
                        BacktestTrade(
                            stock=stock,
                            entry_ts=pos["entry_ts"],
                            entry_price=pos["entry"],
                            exit_ts=nxt["ts"] if reason == "eod" else row["ts"],
                            exit_price=exit_price,
                            qty=pos["qty"],
                            pnl_pct=pnl_pct,
                            reason=reason,
                        )
                    )
                    del positions_bt[stock]

    new_capital = capital + cash_pnl
    return new_capital, trades, cash_pnl


def walk_forward_backtest(data: pd.DataFrame):
    if data.empty:
        raise ValueError("No data")

    dates = sorted(data["ts"].dt.date.unique())
    warmup_days = 20

    capital = INITIAL_CAPITAL
    equity_rows = []
    all_trades = []

    for idx, d in enumerate(dates):
        if idx < warmup_days:
            continue

        train_df = data[data["ts"].dt.date < d].copy()
        day_df = data[data["ts"].dt.date == d].copy()

        if len(train_df) < 500 or len(day_df) < 50:
            continue

        clf = train_model(train_df)
        capital, trades, day_pnl = backtest_one_day(day_df, clf, capital)
        all_trades.extend(trades)
        equity_rows.append({
            "date": d,
            "equity": capital,
            "day_pnl": day_pnl,
        })

    eq = pd.DataFrame(equity_rows)
    tr = pd.DataFrame([t.__dict__ for t in all_trades])

    if len(eq):
        eq["peak"] = eq["equity"].cummax()
        eq["drawdown"] = eq["equity"] / eq["peak"] - 1.0
        max_dd = float(eq["drawdown"].min())
        final_equity = float(eq["equity"].iloc[-1])
        total_ret = final_equity / INITIAL_CAPITAL - 1.0
    else:
        max_dd = 0.0
        final_equity = INITIAL_CAPITAL
        total_ret = 0.0

    summary = {
        "trades": int(len(tr)),
        "win_rate": float((tr["pnl_pct"] > 0).mean()) if len(tr) else 0.0,
        "avg_trade": float(tr["pnl_pct"].mean()) if len(tr) else 0.0,
        "final_equity": final_equity,
        "total_return": total_ret,
        "max_drawdown": max_dd,
    }
    return summary, eq, tr


def run_backtest():
    raw = fetch_universe_history(UNIVERSE, START_DATE, END_DATE)
    if raw.empty:
        raise RuntimeError("No historical data fetched")

    data = build_dataset(raw)
    if data.empty:
        raise RuntimeError("Feature dataset is empty")

    # holdout
    split_point = data["ts"].quantile(0.7)
    train_df = data[data["ts"] <= split_point].copy()
    test_df = data[data["ts"] > split_point].copy()

    clf = train_model(train_df)
    preds = clf.predict_proba(test_df[FEATURE_COLS].fillna(0.0))[:, 1]

    if test_df["label"].nunique() > 1:
        auc = roc_auc_score(test_df["label"], preds)
    else:
        auc = np.nan

    yhat = (preds >= BUY_PROB_THRESHOLD).astype(int)
    print("\n=== Holdout report ===")
    print(classification_report(test_df["label"], yhat, digits=4))
    print(f"AUC: {auc:.4f}" if not np.isnan(auc) else "AUC: N/A")

    summary, eq, tr = walk_forward_backtest(data)

    print("\n=== Walk-forward summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if len(eq):
        eq.to_csv(EQUITY_PATH, index=False, encoding="utf-8-sig")
        print(f"Saved: {EQUITY_PATH}")
    if len(tr):
        tr.to_csv(TRADES_PATH, index=False, encoding="utf-8-sig")
        print(f"Saved: {TRADES_PATH}")

    save_model(clf)
    return clf, summary, eq, tr


# =========================
# Live data handling
# =========================
def update_live_kbar(stock: str, quote: dict):
    """
    官方 tick quote 會提供 Close / Time / Volume / VolSum 之類欄位。
    我們先把 tick 累積成 1 分鐘K。
    """
    ts_str = quote.get("Time")
    close = quote.get("Close")
    volume = quote.get("Volume")

    if ts_str is None or close is None:
        return

    # close / volume may be list
    if isinstance(close, list):
        close = close[0]
    if isinstance(volume, list):
        volume = volume[0] if volume else 0

    now_ts = datetime.now()
    try:
        hh, mm, ss = str(ts_str).split(".")[0].split(":")
        tick_time = now_ts.replace(hour=int(hh), minute=int(mm), second=int(ss), microsecond=0)
    except Exception:
        tick_time = now_ts

    tick_buffer[stock].append({
        "ts": tick_time,
        "close": float(close),
        "volume": float(volume or 0),
    })

    # keep memory bounded
    if len(tick_buffer[stock]) > 3000:
        tick_buffer[stock] = tick_buffer[stock][-1500:]

    df = pd.DataFrame(tick_buffer[stock])
    if df.empty:
        return

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()

    minute = df.resample("1min").agg(
        open=("close", "first"),
        high=("close", "max"),
        low=("close", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()

    if not minute.empty:
        kbar_live[stock] = minute


def latest_signal(stock: str):
    if stock not in kbar_live or model is None:
        return None

    df = kbar_live[stock].reset_index().rename(columns={"index": "ts"})
    df["stock"] = stock
    feat = add_features(df)
    feat = feat.dropna(subset=FEATURE_COLS)
    if feat.empty:
        return None

    row = feat.iloc[-1]
    prob = float(model.predict_proba(row[FEATURE_COLS].fillna(0.0).to_frame().T)[0, 1])

    return {
        "stock": stock,
        "ts": row["ts"],
        "prob": prob,
        "close": float(row["close"]),
        "ma5": float(row["ma5"]) if pd.notna(row["ma5"]) else np.nan,
        "ma20": float(row["ma20"]) if pd.notna(row["ma20"]) else np.nan,
        "vwap": float(row["vwap"]) if pd.notna(row["vwap"]) else np.nan,
        "atr14": float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan,
        "volume_ratio": float(row["volume_ratio"]) if pd.notna(row["volume_ratio"]) else np.nan,
    }


def market_filter():
    sig = latest_signal(MARKET_PROXY)
    if sig is None:
        return True
    return sig["ma5"] > sig["ma20"] and sig["close"] > sig["vwap"]


def can_trade(stock: str):
    if stock not in last_order_time:
        return True
    return (datetime.now() - last_order_time[stock]).seconds > COOLDOWN_SECONDS


def position_size(price: float):
    return max(int(current_capital * RISK_PER_TRADE / price), 1)


def parse_event_payload(stat, msg):
    """
    不同版本的 order callback 物件型態可能不同。
    這裡盡量把它轉成 dict 再判斷。
    """
    s = str(stat)
    m = obj_to_dict(msg)
    text = f"{s} {m}"
    return s, m, text


def extract_deal_info(msg_dict):
    """
    優先從 deal event 抽資料。
    常見欄位：action / code / price / quantity / custom_field
    """
    if not isinstance(msg_dict, dict):
        return None

    # deal event may be nested or flat
    if "trade_id" in msg_dict or ("price" in msg_dict and "quantity" in msg_dict and "code" in msg_dict):
        return {
            "action": msg_dict.get("action"),
            "code": msg_dict.get("code"),
            "price": float(msg_dict.get("price")) if msg_dict.get("price") is not None else None,
            "quantity": int(msg_dict.get("quantity")) if msg_dict.get("quantity") is not None else None,
            "custom_field": msg_dict.get("custom_field"),
            "raw": msg_dict,
        }

    # nested style
    if "order" in msg_dict:
        order = msg_dict.get("order", {}) or {}
        status = msg_dict.get("status", {}) or {}
        contract = msg_dict.get("contract", {}) or {}
        flat = {
            "action": order.get("action"),
            "code": contract.get("code"),
            "price": order.get("price"),
            "quantity": order.get("quantity"),
            "custom_field": order.get("custom_field"),
            "status_id": status.get("id"),
            "raw": msg_dict,
        }
        return flat

    return None


def update_position_from_deal(deal):
    global daily_pnl

    if deal is None:
        return

    stock = deal.get("code")
    action = deal.get("action")
    price = deal.get("price")
    qty = deal.get("quantity")
    tag = deal.get("custom_field")

    if not stock or price is None or qty is None:
        return

    if action == "Buy":
        positions[stock] = Position(
            stock=stock,
            qty=qty,
            entry=price,
            entry_time=datetime.now(),
            order_tag=tag or "",
        )
        log(f"FILLED BUY {stock} @ {price} qty={qty} tag={tag}")

    elif action == "Sell":
        if stock in positions:
            pos = positions[stock]
            pnl_pct = (price - pos.entry) / pos.entry
            daily_pnl += pnl_pct
            del positions[stock]
            log(f"FILLED SELL {stock} @ {price} pnl={pnl_pct:.2%} tag={tag}")


def order_callback(stat, msg):
    s, m, text = parse_event_payload(stat, msg)
    log(f"ORDER EVENT: {s}")

    # try to detect deal / fill
    deal = extract_deal_info(m)
    if deal is not None:
        # only update when action is available
        update_position_from_deal(deal)


api.set_order_callback(order_callback)


def place_stock_order(stock: str, action: str, price: float, qty: int):
    contract = get_contract(stock)
    tag = make_order_tag(stock)

    order = api.Order(
        price=price,
        quantity=qty,
        action=action,
        price_type=sj.constant.StockPriceType.LMT,
        order_type=sj.constant.OrderType.ROD,
        order_lot=sj.constant.StockOrderLot.Common,
        custom_field=tag,
        account=api.stock_account,
    )

    # non-blocking mode; order callback will carry the rest of info
    trade = api.place_order(contract, order, timeout=0)
    pending_orders[tag] = {
        "stock": stock,
        "action": action,
        "price": price,
        "qty": qty,
        "time": datetime.now(),
        "trade": trade,
    }
    return tag, trade


def cancel_stale_orders():
    now = datetime.now()
    for tag, meta in list(pending_orders.items()):
        if (now - meta["time"]).seconds > 10:
            try:
                # cancel_order API signature can vary by version; this is the common pattern
                api.cancel_order(meta["trade"])
                log(f"CANCEL {meta['stock']} tag={tag}")
            except Exception as e:
                log(f"Cancel failed {tag}: {e}")
            finally:
                pending_orders.pop(tag, None)


def score_row(row):
    score = 0.0
    score += 1.2 if row["close"] > row["vwap"] else 0
    score += 1.0 if row["ma5"] > row["ma20"] else 0
    score += 1.5 if row["volume_ratio"] > 1.5 else 0
    score += 2.0 if row["breakout_20"] > 0 else 0
    score += 1.0 if row["pullback_ma5"] > 0 else 0
    score += 0.8 if row["rsi14"] > 50 else 0
    return score


def rank_candidates():
    candidates = []
    for s in UNIVERSE:
        sig = latest_signal(s)
        if sig is None:
            continue

        df = kbar_live[s].reset_index().rename(columns={"index": "ts"})
        df["stock"] = s
        feat = add_features(df).dropna(subset=FEATURE_COLS)
        if feat.empty:
            continue

        row = feat.iloc[-1]
        prob = sig["prob"]
        score = score_row(row) + prob * 3.0

        candidates.append((s, score, sig, row))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:MAX_POSITIONS]


def exit_condition(sig, pos: Position, row):
    if pd.isna(sig["atr14"]):
        return None

    pnl = (sig["close"] - pos.entry) / pos.entry
    stop_loss = -1.5 * sig["atr14"] / pos.entry
    take_profit = 2.2 * sig["atr14"] / pos.entry

    if pnl <= stop_loss:
        return "stop"
    if pnl >= take_profit:
        return "take"

    age_minutes = (datetime.now() - pos.entry_time).total_seconds() / 60.0
    if age_minutes >= HOLDING_MINUTES_MAX:
        return "time"
    return None


def live_trade_logic():
    global daily_pnl

    now = datetime.now().time()
    if now < dtime(9, 5):
        return

    if daily_pnl <= MAX_DAILY_LOSS:
        return

    if not market_filter():
        return

    cancel_stale_orders()

    # exits first
    for stock, pos in list(positions.items()):
        sig = latest_signal(stock)
        if sig is None:
            continue

        if pd.isna(sig["atr14"]):
            continue

        df = kbar_live[stock].reset_index().rename(columns={"index": "ts"})
        df["stock"] = stock
        feat = add_features(df).dropna(subset=FEATURE_COLS)
        if feat.empty:
            continue
        row = feat.iloc[-1]

        reason = exit_condition(sig, pos, row)
        if reason is not None:
            sell_price = round(sig["close"] * 0.998, 2)
            try:
                place_stock_order(stock, "Sell", sell_price, pos.qty)
                log(f"EXIT SIGNAL {stock} reason={reason} pnl={(sig['close']-pos.entry)/pos.entry:.2%}")
            except Exception as e:
                log(f"SELL failed {stock}: {e}")

    # entries
    picks = rank_candidates()
    for stock, score, sig, row in picks:
        if stock in positions:
            continue
        if not can_trade(stock):
            continue

        if pd.isna(sig["atr14"]):
            continue

        # no chase
        if sig["close"] > sig["ma5"] * (1 + BUY_NOT_TOO_EXTENDED):
            continue

        if sig["prob"] < BUY_PROB_THRESHOLD:
            continue

        qty = position_size(sig["close"])
        buy_price = round(sig["close"] * 1.002, 2)

        try:
            place_stock_order(stock, "Buy", buy_price, qty)
            last_order_time[stock] = datetime.now()
            log(f"BUY SIGNAL {stock} prob={sig['prob']:.3f} score={score:.2f} price={buy_price} qty={qty}")
        except Exception as e:
            log(f"BUY failed {stock}: {e}")


# =========================
# Quote callback
# =========================
@api.quote.on_quote
def quote_callback(topic: str, quote: dict):
    # topic example: MKT/redisrd/TSE/2330
    m = re.search(r"/(\d{4,6})$", topic)
    if not m:
        return

    stock = m.group(1)
    if stock not in set(UNIVERSE + [MARKET_PROXY]):
        return

    update_live_kbar(stock, quote)

    # trigger only when enough data exists
    if stock in UNIVERSE or stock == MARKET_PROXY:
        live_trade_logic()


@api.quote.on_event
def quote_event_callback(resp_code: int, event_code: int, info: str, event: str):
    log(f"QUOTE EVENT code={event_code} event={event} info={info}")


# =========================
# Live / Backtest runners
# =========================
def subscribe_live():
    for s in set(UNIVERSE + [MARKET_PROXY]):
        contract = get_contract(s)
        api.quote.subscribe(contract, quote_type=sj.constant.QuoteType.Tick)
        log(f"Subscribed tick: {s}")


def run_live():
    global model
    model = load_model()
    if model is None:
        raise RuntimeError("Model not found. Please run run_backtest() first.")

    subscribe_live()
    log("LIVE started")

    while True:
        time_module.sleep(1)


def run_all():
    """
    先回測 + 訓練模型，然後你可以切到 live。
    """
    clf, summary, eq, tr = run_backtest()
    return clf, summary, eq, tr


if __name__ == "__main__":
    # 先跑回測/訓練
    run_all()

    # 實盤時改成下面這行：
    # run_live()
