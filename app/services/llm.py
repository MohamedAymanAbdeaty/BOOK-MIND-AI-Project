import json
import re


class LLMService:
    """
    A simple service to interact with the Groq Large Language Model (LLM).
    It ensures that we always get a valid JSON response back.
    """

    def __init__(self, api_key: str, model: str, client=None):
        self.api_key = api_key
        self.model = model
        self._client = client

    @property
    def client(self):
        """Lazy-loads the Groq client only when needed."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("GROQ_API_KEY is not configured")
            
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        
        return self._client

    @staticmethod
    def _parse_json(content: str) -> dict:
        """
        Attempts to parse JSON from the model's text response.
        If there's extra text around the JSON (like markdown), it uses a regular
        expression to find just the JSON part and parses that.
        """
        try:
            # Try to parse it directly first
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, look for something that looks like { ... }
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("Model response did not contain valid JSON")
            
            return json.loads(match.group(0))

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict:
        """
        Sends a request to the LLM and guarantees a JSON dictionary in return.
        
        Args:
            system_prompt: The instructions for how the AI should behave.
            user_prompt: The actual question or input from the user.
            temperature: How creative the AI should be (lower = more deterministic).
        """
        try:
            # Ask the model for a response and explicitly request JSON format
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}
                ],
            )
            return self._parse_json(response.choices[0].message.content or "{}")
            
        except Exception as exc:
            # Fallback: Sometimes the server rejects the "json_object" format if it thinks
            # the prompt didn't explicitly ask for JSON. We retry with a stronger prompt.
            if "json_validate_failed" in str(exc) or "400" in str(exc):
                fallback_system_prompt = f"{system_prompt}\nCRITICAL: Respond ONLY with a raw JSON object. Do not include markdown code block formatting or extra commentary."
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": fallback_system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return self._parse_json(response.choices[0].message.content or "{}")
            
            # If it's a different error, we just raise it
            raise exc
