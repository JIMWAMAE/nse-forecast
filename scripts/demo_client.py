"""Example client: calls the running API (make serve or make docker first).

    python scripts/demo_client.py --url http://127.0.0.1:8000
"""
import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd


def call(url, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    a = ap.parse_args()

    print("Health:", call(f"{a.url}/health"))
    print("\nAll stocks (most volatile first):")
    print(pd.DataFrame(call(f"{a.url}/forecast")["forecasts"])
          [["code", "as_of_date", "last_price", "forecast_price", "volatility_1d_pct",
            "range_95_low", "range_95_high"]].to_string(index=False))

    # A client sending its own recent data for Safaricom
    h = pd.read_csv(Path(__file__).resolve().parents[1] / "models" / "history.csv")
    h = h[h["code"] == "SCOM"].tail(60)
    rows = [{"date": r.date, "day_price": r.price, "previous": r.previous, "day_low": r.day_low,
             "day_high": r.day_high, "volume": r.volume} for r in h.itertuples()]
    print("\nPOST /forecast/SCOM with 60 rows of client data:")
    print(json.dumps(call(f"{a.url}/forecast/SCOM", "POST", {"rows": rows}), indent=2))
