"""
NeMo Guardrails wrapper for BookMind.

This class is a thin layer around LLMRails.
- Lazy-loads the rails on first use (NeMo is slow to initialise).
- Exposes a simple synchronous check(question) method.
- Returns a (allowed, code, message) tuple that InputGuard understands.
- Falls back gracefully if NeMo is not installed or the API key is missing.
"""
import os
from pathlib import Path

# Colang bot responses that start with this prefix signal a blocked request.
# Defined in nemo_config/bookmd.co as "GUARD_BLOCKED:<code>: <message>"
_BLOCK_PREFIX = "GUARD_BLOCKED:"

# Path to the NeMo config directory (relative to the project root)
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "nemo_config"


class NemoGuard:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self._rails = None

    @property
    def rails(self):
        """Lazy-loads the LLMRails instance. Called only on first check()."""
        if self._rails is None:
            # NeMo reads GROQ_API_KEY from the environment (set in config.yml as ${GROQ_API_KEY})
            os.environ.setdefault("GROQ_API_KEY", self.groq_api_key)
            from nemoguardrails import LLMRails, RailsConfig
            config = RailsConfig.from_path(str(_CONFIG_DIR))
            self._rails = LLMRails(config)
        return self._rails

    def check(self, question: str) -> tuple[bool, str, str]:
        """
        Ask NeMo Guardrails whether this question is safe to process.

        Returns:
            (allowed, code, message)
            - allowed=True  → the request can continue to the RAG pipeline
            - allowed=False → the request is blocked; code and message say why
        """
        try:
            # rails.generate() is the synchronous wrapper.
            # NeMo auto-applies nest_asyncio so this is safe inside Flask routes.
            response = self.rails.generate(
                messages=[{"role": "user", "content": question}]
            )
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            return _parse_response(content)
        except Exception:
            # If NeMo crashes for any reason (network, misconfiguration, etc.),
            # fail open — let the regex guard in InputGuard handle it instead.
            return True, "allowed", ""


def _parse_response(content: str) -> tuple[bool, str, str]:
    """
    Parse the NeMo response string.
    Blocked messages look like: "GUARD_BLOCKED:prompt_injection: This request was blocked..."
    Anything else is considered allowed.
    """
    if content.startswith(_BLOCK_PREFIX):
        rest = content[len(_BLOCK_PREFIX):]           # "prompt_injection: This request..."
        code, _, message = rest.partition(": ")        # code="prompt_injection", message="This request..."
        return False, code.strip(), message.strip()
    return True, "allowed", ""
