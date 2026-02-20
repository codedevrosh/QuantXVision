import joblib
from core.config import PROPHET_MODEL_DIR


def predict_prophet(stock, years):

    model_path = PROPHET_MODEL_DIR / f"{stock}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Prophet model not found for {stock}\nExpected: {model_path}"
        )

    model = joblib.load(model_path)

    days = years * 365

    future = model.make_future_dataframe(
        periods=days,
        freq="B"   # business days (important for stocks)
    )

    forecast = model.predict(future)

    return (
        forecast["ds"].tail(days),
        forecast["yhat"].tail(days)
    )
