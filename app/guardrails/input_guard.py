import re
from collections import namedtuple

from app.catalog import BOOKS


# Result returned by check() — allowed=True means the request can proceed
GuardDecision = namedtuple("GuardDecision", ["allowed", "code", "message"], defaults=["allowed", ""])

PROHIBITED_ACTION_MESSAGE = (
    "I’m sorry, but I can’t delete or modify books or library data. "
    "BookMind is a read-only assistant; I can help you explore and understand the selected book instead."
)


# ── Regex fallback ────────────────────────────────────────────────────────────
# Used when NeMo Guardrails is not configured (no API key, etc.)

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

# Administrative and destructive requests are never sent to the RAG pipeline.
# Requiring a library-related target avoids blocking ordinary book questions such
# as "What does the author say about removing fear?".
PROHIBITED_ACTION_PATTERNS = [
    r"\b(delete|remove|erase|destroy|purge)\b.{0,50}\b(book|books|library|catalog|collection)\b",
    r"\b(clear|reset|wipe)\b.{0,50}\b(library|catalog|book collection|book data)\b",
    r"\b(add|upload|import|replace|rename|edit|modify|update|change)\b.{0,50}\b(book|books|book content|metadata|library|catalog|collection)\b",
]

_injection_regexes = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_scope_regexes = [re.compile(p, re.IGNORECASE) for p in OUT_OF_SCOPE_PATTERNS]
_prohibited_action_regexes = [re.compile(p, re.IGNORECASE) for p in PROHIBITED_ACTION_PATTERNS]


def _regex_is_injection(text: str) -> bool:
    normalized = " ".join(text.split())
    return any(r.search(normalized) for r in _injection_regexes)


def _regex_is_off_topic(text: str) -> bool:
    return any(r.search(text) for r in _scope_regexes)


def _regex_is_prohibited_action(text: str) -> bool:
    normalized = " ".join(text.split())
    return any(r.search(normalized) for r in _prohibited_action_regexes)


# ── Book-level scope check ────────────────────────────────────────────────────
# Always runs regardless of whether NeMo or regex is used.
# NeMo doesn't know which book is selected, so this stays in Python.

def _check_book_scope(book_id: str, question: str) -> tuple[bool, str]:
    if book_id not in BOOKS:
        return False, "The selected book is not available."
    selected = BOOKS[book_id]["title"].lower()
    for other_id, book in BOOKS.items():
        if other_id != book_id and book["title"].lower() in question.lower() and selected not in question.lower():
            return False, f"That question appears to be about {book['title']}. Please select that book first."
    return True, ""


# ── InputGuard ────────────────────────────────────────────────────────────────

class InputGuard:
    def __init__(self, max_chars: int = 1200, nemo_guard=None):
        self.max_chars = max_chars
        self.nemo = nemo_guard  # NemoGuard instance, or None for regex-only mode

    def check(self, book_id: str, question: str) -> GuardDecision:
        text = question.strip()

        # 1. Fast checks — no LLM needed
        if not text:
            return GuardDecision(False, "invalid_input", "Please enter a question.")
        if len(text) > self.max_chars:
            return GuardDecision(False, "invalid_input", f"Questions must be under {self.max_chars} characters.")

        # 2. Deterministic read-only boundary — always enforced, even if NeMo is
        # unavailable or incorrectly classifies the request.
        if _regex_is_prohibited_action(text):
            return GuardDecision(False, "prohibited_action", PROHIBITED_ACTION_MESSAGE)

        # 3. Content safety — NeMo Guardrails when API key is set, regex otherwise
        if self.nemo:
            allowed, code, message = self.nemo.check(text)
            if not allowed:
                return GuardDecision(False, code, message)
        else:
            if _regex_is_injection(text):
                return GuardDecision(False, "prompt_injection", "This request was blocked by the prompt-injection guard.")
            if _regex_is_off_topic(text):
                return GuardDecision(False, "out_of_scope", "I can only answer questions supported by the selected book.")

        # 4. Book-level scope — always run (NeMo doesn't know which book is selected)
        ok, message = _check_book_scope(book_id, text)
        if not ok:
            return GuardDecision(False, "out_of_scope", message)

        return GuardDecision(True)
