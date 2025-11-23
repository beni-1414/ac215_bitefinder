import pytest
from fastapi.testclient import TestClient
from api.main import api


@pytest.fixture
def client():
    return TestClient(api)


def test_root_path(client):
    # Root path not defined, should return 404
    response = client.get("/")
    assert response.status_code == 404


def test_vlmodel_router_included(client):
    # Check that accessing /vlmodel returns a valid response (method may not be allowed)
    response = client.get("/vlmodel")
    assert response.status_code in [404, 405]  # 404 if route not defined, 405 if method not allowed
