import os
from pathlib import Path

BLOCK_PREFIX = "GUARD_BLOCKED:"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "nemo_config"


class NemoGuard:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self._rails = None

    @property
    def rails(self):
        if self._rails is None:
            os.environ.setdefault("GROQ_API_KEY", self.groq_api_key)

            from nemoguardrails import LLMRails, RailsConfig

            config = RailsConfig.from_path(str(CONFIG_DIR))
            self._rails = LLMRails(config)
        return self._rails

    def check(self, question: str) -> tuple[bool, str, str]:
        try:
            response = self.rails.generate(messages=[{"role": "user", "content": question}])
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            return _parse_response(content)
        except Exception:  # noqa: BLE001 - The regex guard remains available.
            return True, "allowed", ""


def _parse_response(content: str) -> tuple[bool, str, str]:
    if not content.startswith(BLOCK_PREFIX):
        return True, "allowed", ""

    code, separator, message = content[len(BLOCK_PREFIX):].partition(": ")
    if not separator:
        return False, code.strip(), ""
    return False, code.strip(), message.strip()
