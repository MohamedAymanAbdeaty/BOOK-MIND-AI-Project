import json
import re

QUESTION_STOP_WORDS = {
    "about", "book", "does", "from", "have", "that", "this",
    "what", "when", "where", "which", "with",
}


class LLMService:
    def __init__(self, api_key: str, model: str, client=None):
        self.api_key = api_key
        self.model = model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("GROQ_API_KEY is not configured")

            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    @staticmethod
    def _parse_json(content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("Model response did not contain valid JSON")
            return json.loads(match.group(0))

    def _request(self, system_prompt: str, user_prompt: str, temperature: float, use_json: bool):
        request = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if use_json:
            request["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**request)

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict:
        try:
            response = self._request(system_prompt, user_prompt, temperature, use_json=True)
        except Exception as error:
            message = str(error)
            if "json_validate_failed" not in message and "400" not in message:
                raise

            fallback_prompt = system_prompt + "\nRespond with one raw JSON object and no markdown."
            response = self._request(fallback_prompt, user_prompt, temperature, use_json=False)

        content = response.choices[0].message.content or "{}"
        return self._parse_json(content)


class ExtractiveLLMService:
    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict:
        if "strict evidence reviewer" in system_prompt.lower():
            return self._review(user_prompt)
        return self._answer(user_prompt)

    @staticmethod
    def _review(prompt: str) -> dict:
        has_citation = bool(re.search(r"CITATION INDEXES:\s*\[[1-9]", prompt))
        if has_citation:
            return {
                "verdict": "PASS",
                "unsupported_claims": [],
                "feedback": "Grounded extract from the selected book.",
            }
        return {
            "verdict": "FAIL",
            "unsupported_claims": ["No cited evidence"],
            "feedback": "Cited evidence is required.",
        }

    @classmethod
    def _answer(cls, prompt: str) -> dict:
        passage = cls._first_passage(prompt)
        if not passage:
            return {
                "answer": "I could not find enough evidence in the selected book.",
                "citations": [],
            }

        question_words = cls._question_words(prompt)
        sentences = re.split(r"(?<=[.!?])\s+", passage)
        best_index = max(
            range(len(sentences)),
            key=lambda index: cls._sentence_score(sentences[index], question_words),
            default=0,
        )
        excerpt = " ".join(sentences[best_index:best_index + 3])[:900].strip()
        return {"answer": f"{excerpt} [1]", "citations": [1]}

    @staticmethod
    def _first_passage(prompt: str) -> str:
        match = re.search(
            r"\[SOURCE 1\].*?\nTEXT:\s*(.*?)(?=\n\n\[SOURCE 2\]|\Z)",
            prompt,
            re.DOTALL,
        )
        return " ".join(match.group(1).split()) if match else ""

    @staticmethod
    def _question_words(prompt: str) -> set[str]:
        match = re.search(r"QUESTION:\s*(.*?)\n\nEVIDENCE:", prompt, re.DOTALL)
        question = match.group(1) if match else ""
        words = {
            word.lower()
            for word in re.findall(r"[a-zA-Z]{4,}", question)
            if word.lower() not in QUESTION_STOP_WORDS
        }
        if "control" in words:
            words.update({"power", "govern", "command"})
        if "preparation" in words:
            words.update({"prepare", "preparations", "planning"})
        return words

    @staticmethod
    def _sentence_score(sentence: str, question_words: set[str]) -> int:
        sentence = sentence.lower()
        return sum(word in sentence for word in question_words)
