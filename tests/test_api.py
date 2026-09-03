from app import create_app
from app.models.response import ChatResponse


class FakeWorkflow:
    def invoke(self, book_id, question):
        return ChatResponse(
            answer=f"Supported answer for {book_id}: {question}",
            review_verdict="approved",
            pipeline=["test"],
        )


def make_client(settings):
    app = create_app(settings)
    app.config.update(TESTING=True)
    app.extensions["bookmind_workflow"] = FakeWorkflow()
    return app.test_client()


def test_home_page_renders_product(settings):
    response = make_client(settings).get("/")
    assert response.status_code == 200
    assert b"BookMind" in response.data


def test_books_endpoint_returns_three_books(settings):
    payload = make_client(settings).get("/api/books").get_json()
    assert len(payload["books"]) == 3


def test_chat_endpoint_validates_request(settings):
    response = make_client(settings).post("/api/chat", json={"book_id": "invalid id", "question": "x"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_request"


def test_chat_endpoint_returns_typed_response(settings):
    response = make_client(settings).post(
        "/api/chat",
        json={"book_id": "meditations", "question": "What can we control?"},
    )
    assert response.status_code == 200
    assert response.get_json()["review_verdict"] == "approved"


def test_health_endpoint(settings):
    payload = make_client(settings).get("/api/health").get_json()
    assert payload["status"] == "ok"
    assert "corpus_version" in payload
    assert payload["retrieval_mode"] == "qdrant_with_local_fallback"
    assert payload["answer_mode"] == "extractive"
