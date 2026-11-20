import importlib

import pandas as pd
from fastapi.testclient import TestClient


def client(trained, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(trained / "models"))
    import api.main as m
    importlib.reload(m)
    return TestClient(m.app)


def test_health_and_forecasts(trained, monkeypatch):
    with client(trained, monkeypatch) as c:
        assert c.get("/health").json()["status"] == "ok"
        out = c.get("/forecast").json()["forecasts"]
        assert len(out) == 10
        one = c.get("/forecast/scom").json()["forecasts"][0]
        assert one["range_95_low"] < one["forecast_price"] < one["range_95_high"]
        assert c.get("/forecast/XXXX").status_code == 404


def test_post_own_history(trained, monkeypatch):
    h = pd.read_csv(trained / "models" / "history.csv")
    h = h[h["code"] == "KCB"].tail(80)
    rows = [{"date": d, "day_price": p, "previous": pv, "day_low": lo, "day_high": hi, "volume": v}
            for d, p, pv, lo, hi, v in h[["date", "price", "previous", "day_low", "day_high", "volume"]]
            .itertuples(index=False)]
    with client(trained, monkeypatch) as c:
        r = c.post("/forecast/KCB", json={"rows": rows})
        assert r.status_code == 200, r.text
        assert r.json()["forecasts"][0]["code"] == "KCB"
        assert c.post("/forecast/KCB", json={"rows": rows[:5]}).status_code == 422
