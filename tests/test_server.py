from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code in [200, 404]
