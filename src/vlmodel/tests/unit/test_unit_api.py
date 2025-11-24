from api.main import api  # FastAPI app instance


def test_app_instance():
    # Check app instance exists
    assert api is not None
    # Check app title
    assert api.title == "Vision-Language Model Inference API"
