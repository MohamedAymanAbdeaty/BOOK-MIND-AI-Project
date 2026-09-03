from app.rag.prompts import REVIEWER_SYSTEM, format_sources


class ReviewerAgent:
    def __init__(self, llm):
        self.llm = llm

    def review(self, state: dict) -> dict:
        sources = state.get("retrieved_docs", [])
        citations = state.get("citation_indexes", [])

        if not sources or not citations:
            return {
                "reviewer_verdict": "FAIL",
                "reviewer_feedback": "The answer has no sufficiently strong cited evidence.",
                "unsupported_claims": [state.get("draft_answer", "")],
                "pipeline": [
                    *state.get("pipeline", []),
                    "Reviewer rejected unsupported output",
                ],
            }

        prompt = (
            f"QUESTION: {state['question']}\n\n"
            f"DRAFT: {state['draft_answer']}\n\n"
            f"CITATION INDEXES: {citations}\n\n"
            f"EVIDENCE:\n{format_sources(sources)}"
        )
        result = self.llm.complete_json(REVIEWER_SYSTEM, prompt)
        verdict = "PASS" if str(result.get("verdict", "")).upper() == "PASS" else "FAIL"

        return {
            "reviewer_verdict": verdict,
            "reviewer_feedback": str(result.get("feedback", "")).strip(),
            "unsupported_claims": [str(claim) for claim in result.get("unsupported_claims", [])],
            "pipeline": [*state.get("pipeline", []), f"Reviewer verdict: {verdict}"],
        }
