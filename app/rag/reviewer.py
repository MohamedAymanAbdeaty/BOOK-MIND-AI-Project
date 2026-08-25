from app.rag.prompts import REVIEWER_SYSTEM, format_sources


class ReviewerAgent:
    """
    The Reviewer is the second AI agent in the pipeline.
    Its job is to act as a strict editor. It reads the draft answer from the Researcher
    and compares it to the original text. If the answer contains hallucinations or isn't 
    supported by the text, the Reviewer rejects it.
    """

    def __init__(self, llm):
        self.llm = llm

    def review(self, state: dict) -> dict:
        """
        Evaluates the draft answer against the source evidence.
        """
        # If there are no sources or no citations, it's an automatic fail
        if not state.get("retrieved_docs") or not state.get("citation_indexes"):
            return {
                "reviewer_verdict": "FAIL",
                "reviewer_feedback": "The answer has no sufficiently strong cited evidence.",
                "unsupported_claims": [state.get("draft_answer", "")],
                "pipeline": [*state.get("pipeline", []), "Reviewer rejected unsupported output"],
            }

        # Build the prompt with the draft and the evidence
        user_prompt = (
            f"QUESTION: {state['question']}\n\n"
            f"DRAFT: {state['draft_answer']}\n\n"
            f"CITATION INDEXES: {state['citation_indexes']}\n\n"
            f"EVIDENCE:\n{format_sources(state['retrieved_docs'])}"
        )
        
        # Ask the LLM to act as the Reviewer
        result = self.llm.complete_json(REVIEWER_SYSTEM, user_prompt)
        
        # Check if it passed or failed
        verdict = "PASS" if str(result.get("verdict", "")).upper() == "PASS" else "FAIL"
        
        return {
            "reviewer_verdict": verdict,
            "reviewer_feedback": str(result.get("feedback", "")).strip(),
            "unsupported_claims": [str(c) for c in result.get("unsupported_claims", [])],
            "pipeline": [*state.get("pipeline", []), f"Reviewer verdict: {verdict}"],
        }
