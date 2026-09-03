import re

from langgraph.graph import END, START, StateGraph

from app.catalog import BOOKS
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import validate_output
from app.models.response import ChatResponse
from app.rag.researcher import ResearcherAgent
from app.rag.reviewer import ReviewerAgent
from app.rag.state import RAGState
from app.services.cache import RedisCache
from app.services.embedding import EmbeddingService
from app.services.llm import ExtractiveLLMService, LLMService
from app.services.local_retriever import FallbackRetriever, LocalPdfRetriever
from app.services.vector_store import QdrantService

GREETING_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))(?:\s+there)?[!.?]*$",
    re.IGNORECASE,
)

REFUSAL_MESSAGE = (
    "I cannot answer that confidently because the selected book does not "
    "provide enough supporting evidence."
)


class RAGWorkflow:
    def __init__(self, graph, guard: InputGuard, cache: RedisCache):
        self.graph = graph
        self.guard = guard
        self.cache = cache

    def invoke(self, book_id: str, question: str) -> ChatResponse:
        guard_result = self.guard.check(book_id, question)
        if not guard_result.allowed:
            return self._blocked_response(guard_result)

        if GREETING_PATTERN.fullmatch(question.strip()):
            return self._greeting_response(book_id)

        cached_response = self.cache.get(book_id, question)
        if cached_response:
            cached_response["cached"] = True
            cached_response["pipeline"] = ["Approved answer served from cache"]
            return validate_output(cached_response)

        state = self.graph.invoke({
            "book_id": book_id,
            "question": question,
            "revision_count": 0,
            "pipeline": ["Input validated"],
        })
        response = self._build_response(state)

        if response.review_verdict == "approved":
            self.cache.set_approved(book_id, question, response.model_dump(mode="json"))

        return response

    @staticmethod
    def _blocked_response(guard_result) -> ChatResponse:
        return validate_output({
            "answer": guard_result.message,
            "sources": [],
            "review_verdict": "blocked",
            "review_feedback": guard_result.code,
            "cached": False,
            "pipeline": ["Input guard blocked the request"],
        })

    @staticmethod
    def _greeting_response(book_id: str) -> ChatResponse:
        title = BOOKS[book_id]["title"]
        return validate_output({
            "answer": f"Hello! Ask me a question about {title}, and I’ll answer from the book’s text.",
            "sources": [],
            "review_verdict": "approved",
            "review_feedback": "Conversational greeting; no factual claims to verify.",
            "cached": False,
            "pipeline": ["Input validated", "Greeting handled without evidence retrieval"],
        })

    @staticmethod
    def _build_response(state: RAGState) -> ChatResponse:
        approved = state.get("reviewer_verdict") == "PASS"
        sources = state.get("retrieved_docs", [])
        cited_sources = []

        if approved:
            for index in state.get("citation_indexes", []):
                if 1 <= index <= len(sources):
                    cited_sources.append(sources[index - 1])

        return validate_output({
            "answer": state.get("final_answer") or state.get("draft_answer") or REFUSAL_MESSAGE,
            "sources": cited_sources,
            "review_verdict": "approved" if approved else "refused",
            "review_feedback": state.get("reviewer_feedback", ""),
            "cached": False,
            "pipeline": state.get("pipeline", []),
        })


def _compile_graph(researcher: ResearcherAgent, reviewer: ReviewerAgent, max_revisions: int):
    builder = StateGraph(RAGState)

    def next_step(state: RAGState) -> str:
        if state.get("reviewer_verdict") == "PASS":
            return "finalize"
        if state.get("retrieved_docs") and state.get("revision_count", 0) < max_revisions:
            return "revise"
        return "refuse"

    def mark_revision(state: RAGState) -> dict:
        count = state.get("revision_count", 0) + 1
        pipeline = [*state.get("pipeline", []), f"Handoff to researcher (revision {count})"]
        return {"revision_count": count, "pipeline": pipeline}

    def finalize(state: RAGState) -> dict:
        pipeline = [*state.get("pipeline", []), "Grounded answer approved"]
        return {"final_answer": state["draft_answer"], "pipeline": pipeline}

    def refuse(state: RAGState) -> dict:
        pipeline = [*state.get("pipeline", []), "Answer refused after evidence review"]
        return {"final_answer": REFUSAL_MESSAGE, "pipeline": pipeline}

    builder.add_node("researcher", researcher.retrieve)
    builder.add_node("draft", researcher.draft)
    builder.add_node("reviewer", reviewer.review)
    builder.add_node("mark_revision", mark_revision)
    builder.add_node("finalize", finalize)
    builder.add_node("refuse", refuse)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "draft")
    builder.add_edge("draft", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        next_step,
        {"finalize": "finalize", "revise": "mark_revision", "refuse": "refuse"},
    )
    builder.add_edge("mark_revision", "researcher")
    builder.add_edge("finalize", END)
    builder.add_edge("refuse", END)

    return builder.compile()


def _make_retriever(settings):
    local_retriever = LocalPdfRetriever()
    if settings.demo_mode:
        return local_retriever

    embedding = EmbeddingService(settings.embedding_model)
    qdrant = QdrantService(
        settings.qdrant_url,
        settings.qdrant_collection,
        embedding,
        settings.qdrant_api_key,
    )
    return FallbackRetriever(qdrant, local_retriever)


def _make_llm(api_key: str, model: str):
    if api_key:
        return LLMService(api_key, model)
    return ExtractiveLLMService()


def _make_input_guard(settings):
    nemo = None
    if settings.groq_api_key:
        from app.guardrails.nemo_guard import NemoGuard

        nemo = NemoGuard(settings.groq_api_key)
    return InputGuard(settings.max_question_chars, nemo_guard=nemo)


def build_workflow(settings, *, cache=None, retriever=None, llm=None, reviewer_llm=None) -> RAGWorkflow:
    cache = cache or RedisCache(
        settings.redis_url,
        settings.cache_ttl_seconds,
        settings.corpus_version,
    )
    retriever = retriever or _make_retriever(settings)
    llm = llm or _make_llm(settings.groq_api_key, settings.groq_model)
    reviewer_llm = reviewer_llm or _make_llm(
        settings.groq_api_key,
        settings.groq_reviewer_model,
    )

    researcher = ResearcherAgent(
        retriever,
        llm,
        settings.retrieval_top_k,
        settings.retrieval_score_threshold,
    )
    reviewer = ReviewerAgent(reviewer_llm)
    graph = _compile_graph(researcher, reviewer, settings.max_revision_count)

    return RAGWorkflow(graph, _make_input_guard(settings), cache)
