"""
NLP-based deception signal analyzer.
Analyzes transcript for linguistic deception markers.
"""
import re
import spacy
from dataclasses import dataclass
from typing import List, Tuple
import structlog

log = structlog.get_logger()

DISTANCING_WORDS = ["that", "those", "them", "the person", "he", "she", "they"]
NEGATION_PATTERNS = [
    r"\bi did not\b", r"\bi didn't\b", r"\bi never\b",
    r"\bi would never\b", r"\babsolutely not\b", r"\bof course not\b",
]
OVER_EXPLAIN_MARKERS = [
    "because", "therefore", "the reason", "what happened was",
    "let me explain", "what i mean is", "in other words"
]


@dataclass
class NLPResult:
    distancing_language: float = 0.0
    over_explanation_score: float = 0.0
    spontaneous_negation_score: float = 0.0
    contradiction_count: int = 0
    transcript: str = ""


class NLPDetector:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            log.warning("spaCy model not found, run: python -m spacy download en_core_web_sm")
            self.nlp = None

    def analyze(self, transcript: str) -> NLPResult:
        if not transcript.strip():
            return NLPResult(transcript=transcript)

        text_lower = transcript.lower()
        sentences = re.split(r"[.!?]+", transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        word_count = len(transcript.split())

        distancing = self._score_distancing(text_lower, word_count)
        over_explain = self._score_over_explanation(text_lower, sentences)
        negation = self._score_spontaneous_negation(text_lower, sentences)
        contradictions = self._detect_contradictions(sentences)

        return NLPResult(
            distancing_language=distancing,
            over_explanation_score=over_explain,
            spontaneous_negation_score=negation,
            contradiction_count=contradictions,
            transcript=transcript,
        )

    def _score_distancing(self, text: str, word_count: int) -> float:
        if word_count == 0:
            return 0.0
        count = sum(text.count(w) for w in DISTANCING_WORDS)
        # Ratio relative to first-person usage
        first_person = text.count(" i ") + text.count(" me ") + text.count(" my ")
        if first_person == 0:
            return min(count / 5, 1.0)
        ratio = count / (first_person + 1)
        return float(min(ratio / 3.0, 1.0))

    def _score_over_explanation(self, text: str, sentences: List[str]) -> float:
        count = sum(text.count(m) for m in OVER_EXPLAIN_MARKERS)
        sentence_count = max(len(sentences), 1)
        rate = count / sentence_count
        return float(min(rate / 2.0, 1.0))

    def _score_spontaneous_negation(self, text: str, sentences: List[str]) -> float:
        matches = sum(1 for p in NEGATION_PATTERNS if re.search(p, text))
        sentence_count = max(len(sentences), 1)
        return float(min(matches / sentence_count / 0.5, 1.0))

    def _detect_contradictions(self, sentences: List[str]) -> int:
        """
        Simple heuristic: look for polar opposite statements in
        the same response (yes/no, did/didn't, was/wasn't pairs).
        """
        contradictions = 0
        affirm = re.compile(r"\b(i did|i was|i have|i went|i saw|i took)\b")
        deny = re.compile(r"\b(i did not|i wasn't|i haven't|i didn't|i never)\b")

        has_affirm = any(affirm.search(s.lower()) for s in sentences)
        has_deny = any(deny.search(s.lower()) for s in sentences)
        if has_affirm and has_deny:
            contradictions += 1

        return contradictions
