from schema import MarketFeatures, CandleFeatures, TimeseriesSample
import json
import re

MAX_SAMPLES = 3000

def is_valid_market(m: dict, question) -> bool:
    if (m.get("status") != "closed" and
        m.get("status") != "determined" and 
        m.get("status") != "settled" and 
        m.get("status") != "finalized"):
        return False

    if m.get("market_type") != "binary":
        return False
    
    if m.get("mve_collection_ticker"):
        return False
    
    if m.get("mve_selected_legs"):
        return False

    if m.get("result") not in ["yes", "no"]:
        return False
    
    keywords = ["election", "US Elections", "Primaries", "House","International elections","Senate","Governor",'Trump', 'Congress', 'Melania', 'presidential election', 
                'primary election', 'Democratic nominee', 'Republican', "Swing state", "House majority", "Senate majority", "Mayor election", "referendum", "recall election",
                "Government shutdown", "Debt ceiling", "Tax bill", "Helathcare reform","recount", "foreign election", "National security", "ceasefire", "sanction", "nato",
                "NOMINEE", "Prime Minister"
                'SCOTUS & courts', 'Recurring', 'Iran', 'Hormuz', 'Strait', 'government', 'Kash Patel',
                "Attorney", "Cabinet", "Venezuela", "Hegseth", "Americans", "tariffs", "DHS", "citizenship", "voter", "legislation", "immigration", "immigrants",
                "Justice", "Supreme Court", "Senators", "Fed chair", "federal crime", "approval rating", "defense funding", "boycott", "executive order", "Powell", "Commissioner", "Cory Mills",
                "pardon", "embassy", "Truth Social", "Secretary of Labor", "Presidency", "Pam Bondi", "Mamdani", "Kamala Harris", "House of Representatives", "Legislature"]
    
    def contains_keyword(question, keywords):
        q = question.lower()
        for word in keywords:
            pattern = rf"\b{re.escape(word.lower())}\b"
            if re.search(pattern, q):
                return True
        return False

    if contains_keyword(question, keywords):
        return True
    
    return False

def build_resolved_samples(markets_json : list[dict]):
    """
    Builds training samples ONLY from closed/determined/settled  markets.
    """

    samples = []

    for m in markets_json:

        # ONLY TRAIN ON CLEAN BINARY MARKETS
        question = extract_market_question(m)
        if not is_valid_market(m, question):
            continue

        # label: 1 if YES happened, 0 otherwise
        label = 1.0 if m.get("result") == "yes" else 0.0

        yes_price = float(m["yes_ask_dollars"])
        no_price = float(m["no_ask_dollars"])

        samples.append((MarketFeatures(
            market_id=m["ticker"],

            yes_price=yes_price,
            no_price=no_price,

            last_price_dollars=float(m["last_price_dollars"]),
            volume_history=[float(m["volume_fp"] or 0)],

            price_momentum=None,
            volume_weighted_price=None,
            time_to_resolution=0.0,

            rag_query= question,

            label=label
        ), m))
        
        if len(samples) > MAX_SAMPLES:
            break

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

        result = m.get("result", "unresolved")
        close_time = m.get("close_time", "N/A")

        print("=" * 80)
        print(f"TITLE: {title}")
        print(f"TICKER: {ticker}")
        print(f"STATUS: {status} | TYPE: {market_type}")
        print(f"YES PRICE: {yes:.3f} | NO PRICE: {no:.3f}")
        print(f"VOLUME: {volume}")
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

def _parse_candle(raw: dict) -> CandleFeatures:
    p = raw.get("price", {})
    ask = raw.get("yes_ask", {})
    bid = raw.get("yes_bid", {})
    prev = p.get("previous")
    return CandleFeatures(
        end_period_ts=raw["end_period_ts"],
        ds=raw["ds"],
        price_close=float(p.get("close", 0)),
        price_high=float(p.get("high", 0)),
        price_low=float(p.get("low", 0)),
        price_open=float(p.get("open", 0)),
        price_mean=float(p.get("mean", 0)),
        price_previous=float(prev) if prev is not None else None,
        yes_ask_close=float(ask.get("close", 0)),
        yes_ask_high=float(ask.get("high", 0)),
        yes_ask_low=float(ask.get("low", 0)),
        yes_ask_open=float(ask.get("open", 0)),
        yes_bid_close=float(bid.get("close", 0)),
        yes_bid_high=float(bid.get("high", 0)),
        yes_bid_low=float(bid.get("low", 0)),
        yes_bid_open=float(bid.get("open", 0)),
        volume=float(raw.get("volume", 0)),
        open_interest=float(raw.get("open_interest", 0)),
    )


def load_timeseries_samples(path: str) -> list[TimeseriesSample]:
    """
    Loads market_timeseries.json and returns one TimeseriesSample per market.
    Each sample contains the full ordered sequence of hourly candles and the resolved label.
    """
    with open(path) as f:
        data = json.load(f)

    samples = []
    for market_id, candles in data.items():
        if not candles:
            continue
        parsed = [_parse_candle(c) for c in candles]
        samples.append(TimeseriesSample(
            market_id=market_id,
            series_id=candles[0].get("series_id", ""),
            candles=parsed,
            label=float(candles[-1]["label"]),
        ))
    return samples


def write_to_file(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)
        
def append_to_file(data, file):
    with open(file, "a") as f:
        json.dump(data, f, indent=2)


    

