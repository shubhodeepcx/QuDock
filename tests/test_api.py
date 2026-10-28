import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from qudock import api  # noqa: E402
from qudock.screen import ScreenConfig, Screener  # noqa: E402

SUBS = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN", "CC", "OC"]
TOY = pd.DataFrame({
    "smiles": ["c1ccccc1" + s for s in SUBS] + ["C1CCNCC1" + s for s in SUBS]
    + ["c1ccncc1" + s for s in SUBS],
    "label": [1, 0] * 15,
})


@pytest.fixture(scope="module")
def client():
    api.set_screener(Screener(ScreenConfig(n_qubits=2)).fit(TOY, TOY))
    yield TestClient(api.app)
    api.set_screener(None)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_ready"] is True


def test_screen_ranks_and_reports_invalid(client):
    r = client.post("/screen", json={"smiles": ["Oc1ccccc1", "nope", "NC1CCNCC1"], "top": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["invalid"] == ["nope"]
    assert len(body["molecules"]) == 1 and body["molecules"][0]["rank"] == 1
    assert 0 <= body["molecules"][0]["p_active"] <= 1


def test_bad_requests_are_rejected(client):
    assert client.post("/screen", json={"smiles": []}).status_code == 422
    assert client.post("/screen", json={"smiles": ["nope"]}).status_code == 422
