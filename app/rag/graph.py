from langgraph.graph import END, START, StateGraph

from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import validate_output
from app.rag.researcher import ResearcherAgent
from app.rag.reviewer import ReviewerAgent
from app.rag.state import RAGState
from app.models.response import ChatResponse

# Import our refactored services
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.vector_store import QdrantService
from app.services.cache import RedisCache


class RAGWorkflow:
    """
    The main coordinator for answering questions.
    It checks security, checks the cache, and if needed, runs a multi-agent pipeline
    to research and review an answer before sending it to the user.
    """
    def __init__(self, graph, guard: InputGuard, cache: RedisCache, max_revisions: int = 1):
        self.graph = graph
        self.guard = guard
        self.cache = cache
        self.max_revisions = max_revisions

    def invoke(self, book_id: str, question: str) -> ChatResponse:
        """Processes a user question and returns a final answer."""
        
        # Step 1: Security check. Ensure the question is safe and polite.
        decision = self.guard.check(book_id, question)
        if not decision.allowed:
            return validate_output({
                "answer": decision.message,
                "sources": [],
                "review_verdict": "blocked",
                "review_feedback": decision.code,
                "cached": False,
                "pipeline": ["Input guard blocked the request"],
            })

        # Step 2: Cache check. If we answered this exactly before, return the saved answer.
        cached = self.cache.get(book_id, question)
        if cached:
            cached["cached"] = True
            cached["pipeline"] = ["Approved answer served from cache"]
            return validate_output(cached)

        # Step 3: Run the AI Pipeline (Researcher -> Reviewer)
        state = self.graph.invoke(
            {"book_id": book_id, "question": question, "revision_count": 0, "pipeline": ["Input validated"]}
        )
        
        approved = state.get("reviewer_verdict") == "PASS"

        # Step 4: Format the sources that the AI actually used
        cited_sources = []
        if approved:
            for index in state.get("citation_indexes", []):
                # Ensure the index is valid
                if 1 <= index <= len(state.get("retrieved_docs", [])):
                    cited_sources.append(state["retrieved_docs"][index - 1])

        # Step 5: Build the final response payload
        payload = {
            "answer": state.get("final_answer") or state.get("draft_answer") or "I cannot support an answer from this book.",
            "sources": cited_sources if approved else [],
            "review_verdict": "approved" if approved else "refused",
            "review_feedback": state.get("reviewer_feedback", ""),
            "cached": False,
            "pipeline": state.get("pipeline", []),
        }
        
        response = validate_output(payload)
        
        # Step 6: Save good answers to the cache for next time
        if approved:
            self.cache.set_approved(book_id, question, response.model_dump(mode="json"))
            
        return response


def _compile_graph(researcher: ResearcherAgent, reviewer: ReviewerAgent, max_revisions: int):
    """
    Builds the state machine (flowchart) that controls the AI agents.
    It uses LangGraph to define a sequence of steps the AI takes.
    """
    builder = StateGraph(RAGState)

    # Helper function to decide what to do after the reviewer finishes checking the answer
    def retry_or_finish(state: RAGState) -> str:
        if state.get("reviewer_verdict") == "PASS":
            return "finalize"
        if state.get("revision_count", 0) < max_revisions and state.get("retrieved_docs"):
            return "revise"
        return "refuse"

    # Helper function to track how many times we've tried to fix a bad answer
    def mark_revision(state: RAGState) -> dict:
        count = state.get("revision_count", 0) + 1
        return {"revision_count": count, "pipeline": [*state.get("pipeline", []), f"Handoff to researcher (revision {count})"]}

    # Helper function for a successful outcome
    def finalize(state: RAGState) -> dict:
        return {"final_answer": state["draft_answer"], "pipeline": [*state.get("pipeline", []), "Grounded answer approved"]}

    # Helper function if the AI completely fails to find a good answer
    def refuse(state: RAGState) -> dict:
        return {
            "final_answer": "I cannot answer that confidently because the selected book does not provide enough supporting evidence.",
            "pipeline": [*state.get("pipeline", []), "Answer refused after evidence review"],
        }

    # Define the "nodes" (steps in the process)
    builder.add_node("researcher", researcher.retrieve)
    builder.add_node("draft", researcher.draft)
    builder.add_node("reviewer", reviewer.review)
    builder.add_node("mark_revision", mark_revision)
    builder.add_node("finalize", finalize)
    builder.add_node("refuse", refuse)
    
    # Define the "edges" (how we move from one step to the next)
    builder.add_edge(START, "researcher")     # Start by doing research
    builder.add_edge("researcher", "draft")   # Then draft an answer
    builder.add_edge("draft", "reviewer")     # Then have the reviewer check it
    
    # After the reviewer, we use a conditional edge to decide what to do next based on their verdict
    builder.add_conditional_edges(
        "reviewer",
        retry_or_finish,
        {"finalize": "finalize", "revise": "mark_revision", "refuse": "refuse"},
    )
    
    # If we revise, go back to the researcher
    builder.add_edge("mark_revision", "researcher")
    
    # If we finalize or refuse, we are done
    builder.add_edge("finalize", END)
    builder.add_edge("refuse", END)
    
    return builder.compile()


def build_workflow(settings, *, cache=None, retriever=None, llm=None, reviewer_llm=None) -> RAGWorkflow:
    """
    Factory function to wire all the components together.
    It instantiates the services and agents, compiles the graph, and returns the finished workflow.
    """
    cache = cache or RedisCache(settings.redis_url, settings.cache_ttl_seconds, settings.corpus_version)

    if retriever is None:
        embedding = EmbeddingService(settings.embedding_model)
        retriever = QdrantService(
            settings.qdrant_url, settings.qdrant_collection, embedding, settings.qdrant_api_key,
        )
        
    llm = llm or LLMService(settings.groq_api_key, settings.groq_model)
    reviewer_llm = reviewer_llm or LLMService(settings.groq_api_key, settings.groq_reviewer_model)

    # NeMo Guardrails — active when a real API key is present
    nemo = None
    if settings.groq_api_key:
        from app.guardrails.nemo_guard import NemoGuard
        nemo = NemoGuard(settings.groq_api_key)

    researcher = ResearcherAgent(retriever, llm, settings.retrieval_top_k, settings.retrieval_score_threshold)
    reviewer = ReviewerAgent(reviewer_llm)
    
    graph = _compile_graph(researcher, reviewer, settings.max_revision_count)
    return RAGWorkflow(graph, InputGuard(settings.max_question_chars, nemo_guard=nemo), cache, settings.max_revision_count)
