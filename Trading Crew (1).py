"""
Multi-Agent Trading Crew (CrewAI + Gemini + Pydantic + Alpaca Execution)
========================================================================
Implements a three-agent sequential workflow that keeps each agent in check:
1. The Catalyst & Sentiment Analyst (News & SEC filings -> -1.0 to +1.0 score + freshness)
2. The Quantitative Technical Strategist (RSI, EMA alignment, VWAP, ATR -> "No-Chase" filter)
3. The Chief Risk Officer (CRO) (2:1 Reward-to-Risk, <=1% equity risk -> Pydantic JSON schema)

Prerequisites:
    pip install crewai crewai-tools pydantic yfinance pandas numpy alpaca-py langchain-google-genai

Environment Variables:
    export GEMINI_API_KEY="your-gemini-api-key"
    export ALPACA_API_KEY="your-alpaca-api-key"
    export ALPACA_SECRET_KEY="your-alpaca-secret-key"
"""

import os
import json
from typing import Literal, Optional
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool


# =====================================================================
# 1. Pydantic Output Schema for the Chief Risk Officer (CRO)
# =====================================================================
class TradeExecutionDecision(BaseModel):
    """Strict output schema enforced on the Chief Risk Officer (CRO) agent."""
    ticker: str = Field(..., description="Stock ticker symbol evaluated (e.g., AAPL, NVDA).")
    approved: bool = Field(
        ...,
        description="True ONLY if all catalyst, technical, and risk rules pass; otherwise False."
    )
    direction: Literal["BUY", "SELL", "HOLD"] = Field(
        ...,
        description="Trade direction ('BUY' or 'SELL' if approved, 'HOLD' if rejected)."
    )
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Catalyst sentiment score strictly between -1.0 (catastrophic) and +1.0 (explosive bullish)."
    )
    is_catalyst_fresh: bool = Field(
        ...,
        description="True if the catalyst is fresh and unpriced; False if stale or PR fluff."
    )
    is_overextended: bool = Field(
        ...,
        description="True if technicals flag a 'No-Chase' setup (e.g., RSI > 75 on BUY or RSI < 25 on SELL)."
    )
    entry_price: float = Field(..., description="Proposed entry price per share.")
    stop_loss: float = Field(..., description="Hard stop-loss price per share.")
    take_profit: float = Field(..., description="Target take-profit price per share.")
    reward_to_risk_ratio: float = Field(
        ...,
        description="Calculated Reward-to-Risk ratio: |take_profit - entry_price| / |entry_price - stop_loss|."
    )
    qty: int = Field(
        ...,
        ge=0,
        description="Number of shares to trade so total dollar risk <= 1% of account equity (0 if rejected)."
    )
    total_dollar_risk: float = Field(
        ...,
        ge=0.0,
        description="Total dollar risk at stop-loss: qty * |entry_price - stop_loss|."
    )
    risk_pct_of_equity: float = Field(
        ...,
        ge=0.0,
        description="Total dollar risk as a percentage of account equity (must be <= 1.0%)."
    )
    rejection_reason: Optional[str] = Field(
        None,
        description="Detailed reason for rejection if approved is False; None if approved is True."
    )
    rationale: str = Field(
        ...,
        description="Concise audit summary combining catalyst, technical setup, and risk math."
    )


# =====================================================================
# 2. Custom Tools for News/Filings & Quantitative Technical Indicators
# =====================================================================
@tool("Fetch Financial News and SEC Filings")
def fetch_news_and_filings(ticker: str) -> str:
    """
    Fetches raw financial headlines and recent news/filing summaries for a given {ticker}
    so the Catalyst & Sentiment Analyst can score sentiment (-1.0 to +1.0) and freshness.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        news_items = stock.news or []
        if not news_items:
            return json.dumps({
                "ticker": ticker,
                "status": "no_recent_news",
                "message": "No recent news found via yfinance feed."
            })

        formatted_news = []
        for item in news_items[:8]:
            content = item.get("content", item)
            formatted_news.append({
                "title": content.get("title", item.get("title", "Unknown Title")),
                "publisher": content.get("provider", {}).get("displayName", item.get("publisher", "Unknown")),
                "published_at": content.get("pubDate", item.get("providerPublishTime", "Unknown")),
                "summary": content.get("summary", ""),
            })
        return json.dumps({"ticker": ticker, "news_count": len(formatted_news), "headlines": formatted_news}, indent=2)
    except Exception as e:
        return json.dumps({"ticker": ticker, "error": str(e)})


@tool("Python Technical Analysis Tool")
def python_technical_analysis_tool(ticker: str) -> str:
    """
     strictly queries price, volume, and volatility data for {ticker} to compute:
    - RSI (14-period)
    - EMA alignment (EMA 9, EMA 20, EMA 50)
    - VWAP (Volume Weighted Average Price)
    - ATR (14-period Average True Range)
    - Overextended 'No-Chase' check (e.g., RSI > 75 on buy or RSI < 25 on sell)
    """
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np

        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 50:
            return json.dumps({"ticker": ticker, "error": "Insufficient historical data for technical indicators."})

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # EMAs
        ema_9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_14 = float((100 - (100 / (1 + rs))).iloc[-1])

        # ATR (14)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(window=14).mean().iloc[-1])

        # Rolling 20-day Typical Price VWAP proxy (or session VWAP)
        typical_price = (high + low + close) / 3.0
        vwap_20 = float((typical_price * volume).tail(20).sum() / volume.tail(20).sum())

        last_price = float(close.iloc[-1])
        avg_vol_20 = float(volume.tail(20).mean())
        latest_vol = float(volume.iloc[-1])

        bullish_ema_stack = bool(last_price > ema_9 > ema_20 > ema_50)
        bearish_ema_stack = bool(last_price < ema_9 < ema_20 < ema_50)

        metrics = {
            "ticker": ticker,
            "last_price": round(last_price, 2),
            "rsi_14": round(rsi_14, 2),
            "ema_9": round(float(ema_9), 2),
            "ema_20": round(float(ema_20), 2),
            "ema_50": round(float(ema_50), 2),
            "bullish_ema_alignment": bullish_ema_stack,
            "bearish_ema_alignment": bearish_ema_stack,
            "vwap_20d": round(vwap_20, 2),
            "price_above_vwap": bool(last_price > vwap_20),
            "atr_14": round(atr_14, 2),
            "latest_volume": int(latest_vol),
            "avg_volume_20d": int(avg_vol_20),
            "volume_ratio": round(latest_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0.0,
            "no_chase_buy_flag": bool(rsi_14 > 75.0),
            "no_chase_sell_flag": bool(rsi_14 < 25.0),
            "suggested_long_stop_1_5_atr": round(last_price - (1.5 * atr_14), 2),
            "suggested_long_target_3_0_atr": round(last_price + (3.0 * atr_14), 2),
        }
        return json.dumps(metrics, indent=2)
    except Exception as e:
        return json.dumps({"ticker": ticker, "error": str(e)})


# =====================================================================
# 3. CrewAI Setup (Agents, Tasks, and Crew Factory)
# =====================================================================
def build_trading_crew(model_name: str = "gemini/gemini-2.5-pro") -> Crew:
    """Builds and returns the 3-agent CrewAI pipeline configured with Gemini."""
    gemini_llm = LLM(
        model=model_name,
        temperature=0.1,  # Low temperature for cold objectivity and strict risk math
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # -----------------------------------------------------------------
    # Agent 1: The Catalyst & Sentiment Analyst
    # -----------------------------------------------------------------
    catalyst_analyst = Agent(
        role="Financial News & Macro Catalyst Scanner",
        goal=(
            "Filter out market noise and clickbait to identify genuine, high-impact "
            "fundamental catalysts (earnings surprises, FDA approvals, contract wins, "
            "insider buying) for {ticker}."
        ),
        backstory=(
            "You are a cynical former Wall Street news-desk analyst who knows that 90% of "
            "financial headlines are priced-in noise or PR fluff. You read raw news feeds "
            "and SEC filings with cold objectivity, scoring sentiment strictly between "
            "-1.0 (catastrophic) and +1.0 (explosive bullish catalyst) and flagging "
            "whether the news is fresh or stale."
        ),
        tools=[fetch_news_and_filings],
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False,
    )

    # -----------------------------------------------------------------
    # Agent 2: The Quantitative Technical Strategist
    # -----------------------------------------------------------------
    technical_strategist = Agent(
        role="Price Action & Technical Indicator Specialist",
        goal=(
            "Verify whether {ticker}'s chart structure, volume, and momentum indicators "
            "support entering a trade in the direction of the news catalyst."
        ),
        backstory=(
            "You are a disciplined quantitative chartist who does not care about stories "
            "or hype—only price, volume, and volatility. You never guess indicator values; "
            "you strictly query your Python technical analysis tool to inspect RSI, EMA "
            "alignment, VWAP, and Average True Range (ATR). If a stock is already "
            "overextended (e.g., RSI greater than 75 on a buy signal), you ruthlessly "
            "flag it as a 'No-Chase' setup."
        ),
        tools=[python_technical_analysis_tool],
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False,
    )

    # -----------------------------------------------------------------
    # Agent 3: The Chief Risk Officer (CRO)
    # -----------------------------------------------------------------
    chief_risk_officer = Agent(
        role="Portfolio Risk Manager & Execution Gatekeeper",
        goal=(
            "Protect capital at all costs by auditing the proposed trade from the "
            "Sentiment and Technical agents against strict portfolio risk rules before "
            "authorizing execution."
        ),
        backstory=(
            "You are a paranoid, veteran hedge fund Risk Manager. Your default answer is NO. "
            "You only approve a trade if the catalyst is undeniable, the technical entry "
            "isn't overextended, the Reward-to-Risk ratio is at least 2:1, and the total "
            "dollar risk on the stop-loss does not exceed 1% of the account equity."
        ),
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False,
    )

    # -----------------------------------------------------------------
    # Tasks
    # -----------------------------------------------------------------
    catalyst_task = Task(
        description=(
            "Analyze the latest financial news and SEC filings for {ticker} using your tool. "
            "Filter out PR fluff and clickbait. Identify any genuine fundamental catalyst "
            "(earnings surprise, FDA approval, major contract win, insider buying/selling). "
            "Assign a strict sentiment score between -1.0 and +1.0 and explicitly flag whether "
            "the catalyst is FRESH (unpriced, actionable) or STALE (already priced in)."
        ),
        expected_output=(
            "A structured catalyst dossier containing: (1) Headline summary, "
            "(2) Sentiment Score (-1.0 to +1.0), (3) Freshness flag (FRESH vs. STALE), "
            "and (4) Proposed directional bias (BULLISH, BEARISH, or NEUTRAL)."
        ),
        agent=catalyst_analyst,
    )

    technical_task = Task(
        description=(
            "Query the Python Technical Analysis Tool for {ticker} and audit whether price "
            "action, volume, and volatility support the directional bias from the Catalyst Analyst. "
            "Report exact values for Last Price, RSI(14), EMA(9/20/50) alignment, VWAP, and ATR(14). "
            "If RSI > 75 on a buy setup (or RSI < 25 on a short setup), ruthlessly flag it as a "
            "'No-Chase' overextended setup. Propose an ATR-based entry price, stop-loss, and take-profit."
        ),
        expected_output=(
            "A quantitative technical report listing exact RSI, EMA 9/20/50, VWAP, Volume ratio, "
            "and ATR values, an explicit 'No-Chase' overextension verdict, and concrete entry, "
            "stop-loss, and take-profit levels."
        ),
        agent=technical_strategist,
        context=[catalyst_task],
    )

    risk_audit_task = Task(
        description=(
            "Audit the Catalyst Dossier and Technical Report for {ticker} against our portfolio "
            "account equity of ${account_equity}. Your default answer is NO (approved=False). "
            "Only approve (approved=True) if ALL four conditions hold:\n"
            "  1. Catalyst is undeniable and FRESH (e.g., |sentiment_score| >= 0.6 and is_catalyst_fresh=True).\n"
            "  2. Technical setup aligns with the catalyst and is NOT flagged as 'No-Chase' (is_overextended=False).\n"
            "  3. Reward-to-Risk ratio is at least 2.0 (|take_profit - entry_price| / |entry_price - stop_loss| >= 2.0).\n"
            "  4. Position size (qty) is calculated so total dollar risk (qty * |entry_price - stop_loss|) "
            "does NOT exceed 1% of ${account_equity} (max risk = 0.01 * ${account_equity}).\n"
            "If any rule fails, set approved=False, direction='HOLD', qty=0, and state the exact rejection_reason."
        ),
        expected_output="A strictly validated TradeExecutionDecision JSON object matching the Pydantic schema.",
         output_pydantic=TradeExecutionDecision,
        agent=chief_risk_officer,
        context=[catalyst_task, technical_task],
    )

    return Crew(
        agents=[catalyst_analyst, technical_strategist, chief_risk_officer],
        tasks=[catalyst_task, technical_task, risk_audit_task],
        process=Process.sequential,
        verbose=True,
    )


# =====================================================================
# 4. Alpaca Execution Bridge (Consumes the Pydantic Decision Object)
# =====================================================================
def execute_with_alpaca(decision: TradeExecutionDecision, paper: bool = True) -> dict:
    """
    Connects the CRO's Pydantic output directly to the Alpaca Trading API
    using a bracket order (Entry + Stop-Loss + Take-Profit).
    """
    if not decision.approved or decision.qty <= 0 or decision.direction == "HOLD":
        return {
            "status": "SKIPPED",
            "reason": decision.rejection_reason or "CRO did not approve the trade.",
            "ticker": decision.ticker,
        }

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return {
            "status": "DRY_RUN_ONLY",
            "message": "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Returning validated bracket order payload.",
            "order_payload": {
                "symbol": decision.ticker,
                "qty": decision.qty,
                "side": decision.direction.lower(),
                "order_class": "bracket",
                "stop_loss": {"stop_price": decision.stop_loss},
                "take_profit": {"limit_price": decision.take_profit},
            },
        }

    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
    side = OrderSide.BUY if decision.direction == "BUY" else OrderSide.SELL

    bracket_order = MarketOrderRequest(
        symbol=decision.ticker,
        qty=decision.qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(decision.take_profit, 2)),
        stop_loss=StopLossRequest(stop_price=round(decision.stop_loss, 2)),
    )
    order_response = client.submit_order(order_data=bracket_order)
    return {
        "status": "SUBMITTED",
        "order_id": str(order_response.id),
        "symbol": order_response.symbol,
        "qty": str(order_response.qty),
        "side": str(order_response.side),
    }


# =====================================================================
# 5. CLI Entrypoint
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the 3-Agent Trading Crew (CrewAI + Gemini + Pydantic)")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker symbol to analyze (e.g., NVDA)")
    parser.add_argument("--equity", type=float, default=100000.0, help="Total portfolio account equity in USD")
    parser.add_argument("--execute", action="store_true", help="Submit approved trade to Alpaca Paper API")
    args = parser.parse_args()

    crew = build_trading_crew()
    result = crew.kickoff(inputs={
        "ticker": args.ticker.upper(),
        "account_equity": args.equity,
    })

    # Access the strongly-typed Pydantic model from the CRO task
    cro_decision: TradeExecutionDecision = result.pydantic
    print("\n=== CRO PYDANTIC DECISION ===")
    print(cro_decision.model_dump_json(indent=2))

    if args.execute:
        exec_status = execute_with_alpaca(cro_decision, paper=True)
        print("\n=== ALPACA EXECUTION RESULT ===")
        print(json.dumps(exec_status, indent=2))