import re
from dataclasses import dataclass

from app.catalog import BOOKS

PROHIBITED_ACTION_MESSAGE = (
    "I’m sorry, but I can’t delete or modify books or library data. "
    "BookMind is a read-only assistant; I can help you explore and "
    "understand the selected book instead."
)

INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
    r"\b(reveal|show|print|repeat)\s+(the\s+|your\s+)?(system|developer)\s+prompt\b",
    r"\bforget\s+(your|all|the)\s+(rules|instructions|prompt)\b",
    r"\bpretend\s+(you('| a)?re|to be)\s+(unrestricted|dan|developer)\b",
    r"\b(database|api|secret|access)\s+(key|credentials?|token|password)\b",
    r"\b(jailbreak|prompt injection|bypass (the )?guardrails?)\b",
    r"\bexecute\s+(this|the following)\s+(command|code|instruction)\b",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\b(write|debug|run)\s+(me\s+)?(python|javascript|java|c\+\+|code)\b",
    r"\b(world cup|weather|stock price|bitcoin price|latest news)\b",
    r"\bbook (a |an )?(flight|hotel|restaurant)\b",
]

PROHIBITED_ACTION_PATTERNS = [
    r"\b(delete|remove|erase|destroy|purge)\b.{0,50}\b(book|books|library|catalog|collection)\b",
    r"\b(clear|reset|wipe)\b.{0,50}\b(library|catalog|book collection|book data)\b",
    r"\b(add|upload|import|replace|rename|edit|modify|update|change)\b.{0,50}\b(book|books|book content|metadata|library|catalog|collection)\b",
]


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


INJECTION_RULES = _compile_patterns(INJECTION_PATTERNS)
OUT_OF_SCOPE_RULES = _compile_patterns(OUT_OF_SCOPE_PATTERNS)
PROHIBITED_ACTION_RULES = _compile_patterns(PROHIBITED_ACTION_PATTERNS)


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    code: str = "allowed"
    message: str = ""


def _matches_any(text: str, rules: list[re.Pattern]) -> bool:
    normalized = " ".join(text.split())
    return any(rule.search(normalized) for rule in rules)


def _check_book_scope(book_id: str, question: str) -> GuardDecision | None:
    if book_id not in BOOKS:
        return GuardDecision(False, "out_of_scope", "The selected book is not available.")

    selected_title = BOOKS[book_id]["title"].lower()
    question = question.lower()

    for other_id, book in BOOKS.items():
        other_title = book["title"].lower()
        asks_about_other_book = other_id != book_id and other_title in question
        if asks_about_other_book and selected_title not in question:
            message = f"That question appears to be about {book['title']}. Please select that book first."
            return GuardDecision(False, "out_of_scope", message)
    return None


class InputGuard:
    def __init__(self, max_chars: int = 1200, nemo_guard=None):
        self.max_chars = max_chars
        self.nemo = nemo_guard

    def check(self, book_id: str, question: str) -> GuardDecision:
        text = question.strip()

        if not text:
            return GuardDecision(False, "invalid_input", "Please enter a question.")
        if len(text) > self.max_chars:
            message = f"Questions must be under {self.max_chars} characters."
            return GuardDecision(False, "invalid_input", message)
        if _matches_any(text, PROHIBITED_ACTION_RULES):
            return GuardDecision(False, "prohibited_action", PROHIBITED_ACTION_MESSAGE)

        safety_result = self._check_safety(text)
        if safety_result:
            return safety_result

        scope_result = _check_book_scope(book_id, text)
        return scope_result or GuardDecision(True)

    def _check_safety(self, text: str) -> GuardDecision | None:
        if _matches_any(text, INJECTION_RULES):
            message = "This request was blocked by the prompt-injection guard."
            return GuardDecision(False, "prompt_injection", message)
        if _matches_any(text, OUT_OF_SCOPE_RULES):
            message = "I can only answer questions supported by the selected book."
            return GuardDecision(False, "out_of_scope", message)

        if self.nemo:
            allowed, code, message = self.nemo.check(text)
            if not allowed:
                return GuardDecision(False, code, message)
        return None
