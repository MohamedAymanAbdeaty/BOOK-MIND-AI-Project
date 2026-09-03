from app.rag.prompts import RESEARCHER_SYSTEM, format_sources


class ResearcherAgent:
    def __init__(self, retriever, llm, top_k: int, score_threshold: float):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, state: dict) -> dict:
        sources = self.retriever.search(
            state["book_id"],
            state["question"],
            limit=self.top_k,
            score_threshold=self.score_threshold,
        )
        pipeline = [*state.get("pipeline", []), f"Retrieved {len(sources)} passages"]

        return {
            "retrieved_docs": [source.model_dump() for source in sources],
            "pipeline": pipeline,
        }

    def draft(self, state: dict) -> dict:
        sources = state.get("retrieved_docs", [])
        if not sources:
            return {
                "draft_answer": "I could not find enough evidence in the selected book to answer that question.",
                "citation_indexes": [],
                "pipeline": [*state.get("pipeline", []), "Evidence threshold not met"],
            }

        prompt = self._build_prompt(state, sources)
        result = self.llm.complete_json(RESEARCHER_SYSTEM, prompt)
        citations = self._valid_citations(result.get("citations", []), len(sources))

        return {
            "draft_answer": str(result.get("answer", "")).strip(),
            "citation_indexes": citations,
            "pipeline": [
                *state.get("pipeline", []),
                "Researcher drafted an evidence-bound answer",
            ],
        }

    @staticmethod
    def _build_prompt(state: dict, sources: list[dict]) -> str:
        prompt = f"QUESTION: {state['question']}\n\nEVIDENCE:\n{format_sources(sources)}"
        feedback = state.get("reviewer_feedback")
        if feedback:
            prompt += f"\nREVIEW FEEDBACK FROM THE PREVIOUS DRAFT:\n{feedback}\nRevise conservatively."
        return prompt

    @staticmethod
    def _valid_citations(citations, source_count: int) -> list[int]:
        valid = set()
        for citation in citations:
            if str(citation).isdigit():
                index = int(citation)
                if 1 <= index <= source_count:
                    valid.add(index)
        return sorted(valid)
