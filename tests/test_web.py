# tests/test_web.py
import pytest
import httpx
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Skip all web tests in CI if no checkpoint exists
CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model_v3.pt'))
skip_no_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT),
    reason="No model checkpoint found — skipping web tests in CI"
)

from web.app import app

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c

@pytest.mark.anyio
async def test_index_route(client):
    res = await client.get("/")
    assert res.status_code == 200
    assert "ECABSD" in res.text

@skip_no_checkpoint
@pytest.mark.anyio
async def test_predict_route(client):
    pdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample', '1AY7.pdb'))
    assert os.path.exists(pdb_path)
    with open(pdb_path, "rb") as f:
        files = {"pdb_file": ("1AY7.pdb", f, "application/octet-stream")}
        data = {"chain_a": "A", "chain_b": "B", "threshold": "auto", "mode": "threshold"}
        res = await client.post("/predict", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()
    assert json_data["status"] == "success"
    assert "heatmap_url" in json_data
    assert "residues" in json_data
    assert "gradcam_allowed" in json_data
    assert len(json_data["residues"]) > 0
    assert "probability" in json_data["residues"][0]

@skip_no_checkpoint
@pytest.mark.anyio
async def test_explain_route(client):
    pdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample', '1AY7.pdb'))
    assert os.path.exists(pdb_path)
    with open(pdb_path, "rb") as f:
        files = {"pdb_file": ("1AY7.pdb", f, "application/octet-stream")}
        data = {"chain_a": "A", "chain_b": "B", "threshold": "0.5819"}
        res = await client.post("/explain", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()
    assert json_data["status"] == "success"
    assert "gradcam_image" in json_data
    assert "attention_image" in json_data
    assert "overlap_percentage" in json_data

@skip_no_checkpoint
@pytest.mark.anyio
async def test_explain_route_low_memory(client, monkeypatch):
    import web.app
    monkeypatch.setenv("IS_RENDER", "true")
    monkeypatch.setattr(web.app, "has_enough_memory", lambda min_free_mb=250: (False, 120.0))
    pdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample', '1AY7.pdb'))
    assert os.path.exists(pdb_path)
    with open(pdb_path, "rb") as f:
        files = {"pdb_file": ("1AY7.pdb", f, "application/octet-stream")}
        data = {"chain_a": "A", "chain_b": "B", "threshold": "0.5819"}
        res = await client.post("/explain", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()
    assert json_data["status"] == "success"
    assert json_data["gradcam_available"] is False
    assert "skipped" in json_data["gradcam_message"]
    assert json_data["gradcam_image"] is None
    assert json_data["attention_image"] is not None
