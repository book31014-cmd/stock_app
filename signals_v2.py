"""
signals_v2.py
- 分數可解釋（score + breakdown）
- RR 風報比納入決策
- 行動分類更細（STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL）
- 停損改為「結構型」：優先 BB_LOW，其次 MA50*0.97（保底）

⚠️ 本檔案假設 indicators.add_indicators 會產生以下欄位（你的程式已用到）：
MA20, MA50, RSI14, BB_UP, BB_LOW
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union


@dataclass
class SignalConfig:
    # 你原本可能已有更多參數；先保留擴充彈性
    rsi_buy: float = 35.0
    rsi_sell: float = 70.0
    score_strong_buy: int = 80
    score_buy: int = 65
    score_sell: int = 40
    rr_buy: float = 1.3
    rr_strong_buy: float = 2.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def suggest_price_levels(today_row) -> Dict[str, float]:
    """
    回傳建議價位 dict：buy_low, buy_high, sell_ref, stop_loss
    - sell_ref: BB_UP（布林上軌）
    - buy 區間：BB_LOW 到 MA20（偏保守），若缺欄位則退回 MA20*0.98 ~ MA20*1.0
    - stop_loss：結構型：min(BB_LOW, MA50*0.97)；若 BB_LOW 缺則只用 MA50*0.97
    """
    ma20 = float(today_row.get("MA20"))
    ma50 = float(today_row.get("MA50"))
    bb_up = float(today_row.get("BB_UP"))
    bb_low = float(today_row.get("BB_LOW"))

    # 買入區：偏保守（靠近下緣）
    buy_low = bb_low
    buy_high = min(ma20, bb_low * 1.06)  # 不要拉太高，避免追

    # 賣出參考：上軌
    sell_ref = bb_up

    # 停損：結構型（優先下軌，保底用 MA50*0.97）
    ma50_sl = ma50 * 0.97
    stop_loss = min(bb_low, ma50_sl) if bb_low > 0 else ma50_sl

    return {
        "buy_low": float(buy_low),
        "buy_high": float(buy_high),
        "sell_ref": float(sell_ref),
        "stop_loss": float(stop_loss),
    }


def risk_reward(price: Dict[str, Any], close: float) -> float | None:
    """(sell_ref - close) / (close - stop_loss)"""
    try:
        sell_ref = float(price["sell_ref"])
        stop_loss = float(price["stop_loss"])
        denom = close - stop_loss
        if denom <= 0:
            return None
        rr = (sell_ref - close) / denom
        if rr <= 0:
            return None
        return rr
    except Exception:
        return None


def score_row(today_row, close: float, cfg: SignalConfig | None = None) -> Tuple[int, Dict[str, int]]:
    """
    分數 + 可解釋 breakdown
    目標：0~100

    breakdown:
    - trend（0~35）: Close 相對 MA20/MA50
    - momentum（0~30）: RSI
    - volatility（0~25）: Close 在布林通道位置（越靠近下軌越加分）
    - risk（0~10）: RR 越高越加分
    """
    cfg = cfg or SignalConfig()

    ma20 = float(today_row.get("MA20"))
    ma50 = float(today_row.get("MA50"))
    rsi = float(today_row.get("RSI14"))
    bb_up = float(today_row.get("BB_UP"))
    bb_low = float(today_row.get("BB_LOW"))

    # trend：越低於 MA20/MA50 越偏「便宜」→加分；但太弱也要扣（用 ma50 兜底）
    dist20 = (ma20 - close) / ma20  # 正值代表低於 MA20
    dist50 = (ma50 - close) / ma50
    trend_raw = 0.6 * dist20 + 0.4 * dist50
    trend = int(round(_clamp((trend_raw + 0.02) / 0.12, 0, 1) * 35))  # 約 -2%~10% 映射

    # momentum：RSI 越低越加分（偏逆勢抓低）
    # 以 25~70 做映射
    momentum = int(round(_clamp((70 - rsi) / 45, 0, 1) * 30))

    # volatility：靠近下軌加分，靠近上軌扣分
    width = max(1e-9, bb_up - bb_low)
    pos = (close - bb_low) / width  # 0=下軌，1=上軌
    volatility = int(round(_clamp((1 - pos), 0, 1) * 25))

    # risk：用你自己的價位建議計算 RR
    price = suggest_price_levels(today_row)
    rr = risk_reward(price, close)
    if rr is None:
        risk = 0
    else:
        # RR 1.0~3.0 映射到 0~10
        risk = int(round(_clamp((rr - 1.0) / 2.0, 0, 1) * 10))

    total = int(_clamp(trend + momentum + volatility + risk, 0, 100))

    detail = {
        "trend": trend,
        "momentum": momentum,
        "volatility": volatility,
        "risk": risk,
    }
    return total, detail


def classify_action(score: int, price: Dict[str, Any], close: float, cfg: SignalConfig | None = None) -> str:
    """
    行動分類（更細）
    - STRONG_BUY：score>=80 且 RR>=2 且 close 在買入區間附近
    - BUY：score>=65 且 RR>=1.3 且 close<=buy_high
    - HOLD：其餘
    - SELL：score<40 或 close 跌破 stop_loss
    - STRONG_SELL：close 跌破 stop_loss 且 score 很低
    """
    cfg = cfg or SignalConfig()

    buy_low = float(price["buy_low"])
    buy_high = float(price["buy_high"])
    stop_loss = float(price["stop_loss"])

    rr = risk_reward(price, close) or 0.0

    # 風控優先
    if close <= stop_loss:
        return "STRONG_SELL" if score <= (cfg.score_sell - 10) else "SELL"

    in_buy_zone = (close >= buy_low) and (close <= buy_high * 1.01)

    if score >= cfg.score_strong_buy and rr >= cfg.rr_strong_buy and in_buy_zone:
        return "STRONG_BUY"
    if score >= cfg.score_buy and rr >= cfg.rr_buy and close <= buy_high * 1.01:
        return "BUY"
    if score < cfg.score_sell:
        return "SELL"
    return "HOLD"
