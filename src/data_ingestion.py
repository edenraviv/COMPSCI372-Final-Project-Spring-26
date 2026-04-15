from schema import MarketFeatures
from datetime import datetime, timezone

def is_clean_binary_market(m: dict) -> bool:
    if (m.get("status") != "closed" and
        m.get("status") != "determined" and 
        m.get("status") != "settled"):
        return False

    if m.get("market_type") != "binary":
        return False

    if m.get("result") not in ["yes", "no"]:
        return False

    return True


def build_resolved_samples(markets_json):
    """
    Builds training samples ONLY from closed/determined/settled  markets.
    IMPORTANT: assumes API returns final resolved state.
    """

    samples = []

    for m in markets_json["markets"]:

        # ONLY TRAIN ON CLEAN BINARY MARKETS
        if not is_clean_binary_market(m):
            continue

        # label: 1 if YES happened, 0 otherwise
        label = 1.0 if m.get("result") == "yes" else 0.0

        yes_price = float(m["yes_ask_dollars"])
        no_price = float(m["no_ask_dollars"])

        samples.append(MarketFeatures(
            market_id=m["ticker"],

            yes_price=yes_price,
            no_price=no_price,

            price_history=[yes_price],      # placeholder (no leakage-safe history yet)
            volume_history=[float(m["volume_fp"] or 0)],

            open_interest=float(m["open_interest_fp"] or 0),

            price_momentum=None,
            volume_weighted_price=None,
            time_to_resolution=0.0,

            rag_query=extract_market_question(m),

            label=label
        ))

    return samples

def print_markets(markets):
    for m in markets["markets"][:10]:

        title = m.get("title", "N/A")
        ticker = m.get("ticker", "N/A")
        status = m.get("status", "N/A")
        market_type = m.get("market_type", "N/A")

        yes = float(m.get("yes_ask_dollars", 0))
        no = float(m.get("no_ask_dollars", 0))

        volume = float(m.get("volume_fp", 0))
        open_interest = float(m.get("open_interest_fp", 0))

        result = m.get("result", "unresolved")
        close_time = m.get("close_time", "N/A")

        print("=" * 80)
        print(f"TITLE: {title}")
        print(f"TICKER: {ticker}")
        print(f"STATUS: {status} | TYPE: {market_type}")
        print(f"YES PRICE: {yes:.3f} | NO PRICE: {no:.3f}")
        print(f"VOLUME: {volume} | OPEN INTEREST: {open_interest}")
        print(f"CLOSE TIME: {close_time}")
        print(f"RESULT: {result}")

def extract_market_question(m: dict) -> str:
    """
    Builds a clean, human-readable question from Kalshi market data.
    """

    #try rules
    rules = m.get("rules_primary", "").strip()
    if rules:
        return rules
    
    # try subtitles 
    yes_sub = m.get("yes_sub_title")
    no_sub = m.get("no_sub_title")

    if yes_sub:
        return yes_sub
    
    if no_sub:
        return no_sub

    # fallback to title (deprecated)
    title = m.get("title", "")

    # 3. fallback to custom strike structure
    custom = m.get("custom_strike", {})
    associated = custom.get("Associated Markets", "")

    if associated:
        return associated

    return title