from app.rag.prompts import RESEARCHER_SYSTEM, format_sources


class ResearcherAgent:
    """
    The Researcher is the first AI agent in the pipeline.
    Its job is to search the book for relevant passages and draft an answer based ONLY on those passages.
    """

    def __init__(self, retriever, llm, top_k: int, score_threshold: float):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k  # Maximum number of passages to fetch
        self.score_threshold = score_threshold  # Minimum relevance score needed

    def retrieve(self, state: dict) -> dict:
        """
        Step 1: Fetch relevant information from the database.
        """
        sources = self.retriever.search(
            state["book_id"],
            state["question"],
            limit=self.top_k,
            score_threshold=self.score_threshold,
        )
        
        # Keep track of what we did for debugging
        pipeline_log = [*state.get("pipeline", []), f"Retrieved {len(sources)} passages"]
        
        return {
            "retrieved_docs": [source.model_dump() for source in sources], 
            "pipeline": pipeline_log
        }

    def draft(self, state: dict) -> dict:
        """
        Step 2: Read the retrieved passages and draft an answer.
        """
        sources = state.get("retrieved_docs", [])
        
        # If we found nothing useful, we give up early
        if not sources:
            return {
                "draft_answer": "I could not find enough evidence in the selected book to answer that question.",
                "citation_indexes": [],
                "pipeline": [*state.get("pipeline", []), "Evidence threshold not met"],
            }

        # If the Reviewer rejected our previous draft, we get their feedback here so we can do better
        feedback = state.get("reviewer_feedback", "")
        revision_note = ""
        if feedback:
            revision_note = f"\nREVIEW FEEDBACK FROM THE PREVIOUS DRAFT:\n{feedback}\nRevise conservatively."
            
        # Build the prompt to send to the LLM
        user_prompt = (
            f"QUESTION: {state['question']}\n\n"
            f"EVIDENCE:\n{format_sources(sources)}"
            f"{revision_note}"
        )
        
        # Ask the LLM to write the draft
        result = self.llm.complete_json(RESEARCHER_SYSTEM, user_prompt)
        
        # Extract the citation numbers the LLM used and ensure they are valid
        raw_citations = result.get("citations", [])
        citations = sorted({
            int(i) for i in raw_citations 
            if str(i).isdigit() and 1 <= int(i) <= len(sources)
        })
        
        return {
            "draft_answer": str(result.get("answer", "")).strip(),
            "citation_indexes": citations,
            "pipeline": [*state.get("pipeline", []), "Researcher drafted an evidence-bound answer"],
        }
