import os
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd

MODEL_PATH = Path(__file__).parent / "prediction_model/price_recommendation_model.pkl"

_MODEL = None

def get_model():
    """Load the model once into memory."""
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            return None
        with MODEL_PATH.open("rb") as f:
            _MODEL = pickle.load(f)
    return _MODEL


def predict_buying_decision(
    current_price: float,
    lowest_price_so_far: float,
    category: str = "smartphone",
    store: str = "Amazon India",
    avg_price: float | None = None,
) -> dict:
    """Calculates feature values and generates a model-backed verdict."""
    model = get_model()
    now = datetime.now()

    # Approximate rolling stats from available scrapers if daily history isn't continuous
    base_reference = avg_price if avg_price and avg_price > 0 else current_price
    rolling_7d = (current_price + base_reference) / 2
    rolling_30d = base_reference
    price_7d_ago = rolling_7d
    price_30d_ago = base_reference

    # Calculate discount from 30d reference
    discount = 0.0
    if base_reference > current_price:
        discount = round(((base_reference - current_price) / base_reference) * 100, 2)

    # Detect Indian festival sale periods (September - November)
    month = now.month
    festival_flag = 1 if month in [9, 10, 11] else 0

    features_dict = {
        "current_price": [current_price],
        "price_7_days_ago": [price_7d_ago],
        "price_30_days_ago": [price_30d_ago],
        "rolling_mean_7_days": [rolling_7d],
        "rolling_mean_30_days": [rolling_30d],
        "lowest_price_so_far": [min(current_price, lowest_price_so_far)],
        "discount_percent": [discount],
        "month": [month],
        "day_of_week": [now.weekday()],
        "festival_sale_flag": [festival_flag],
        "category": [category],
        "store": [store],
    }

    df_features = pd.DataFrame(features_dict)

    if model is None:
        # Fallback heuristic if .pkl is missing
        drop_prob = 0.8 if current_price > (lowest_price_so_far * 1.1) else 0.2
    else:
        # 1 means wait for drop >= 5% in next 30 days
        drop_prob = float(model.predict_proba(df_features)[0, 1])

    # Convert probability to actionable decision badge
    if drop_prob >= 0.65:
        verdict = "🔴 WAIT"
        action = "High probability of a ≥5% price drop within the next 30 days."
    elif drop_prob <= 0.35:
        verdict = "🟢 BUY NOW"
        action = "Price is near its baseline floor; low probability of a major drop."
    else:
        verdict = "🟡 FAIR DEAL"
        action = "Moderate pricing; buy if needed immediately, or monitor for minor bank offers."

    return {
        "verdict": verdict,
        "wait_probability": round(drop_prob * 100, 1),
        "action_advice": action,
        "discount_detected": f"{discount}%",
    }