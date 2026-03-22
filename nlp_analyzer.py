"""
NLPAnalyzer — linguistic deception indicators from transcript:
- Distancing language ("that person" vs "I")
- Spontaneous negations ("I did NOT do it")
- Over-explanation / unnecessary detail
- Contradictions across the conversation
- Cognitive load markers (complex phrasing when simple suffices)
- Negative emotion words
- Exclusive words (but, except, without) — liars use fewer
"""
import asyncio
import re
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class NLPResult:
    deception_score: float
    confidence: float
    distancing_score: float
    negation_rate: float
    over_explanation_score: float
    contradiction_count: int
    cognitive_load_score: float
    flags: List[str]


class NLPAnalyzer:
    FIRST_PERSON = {"i", "me", "my", "mine", "myself"}
    DISTANCING = {"that person", "he", "she", "they", "him", "her", "them", "it"}
    NEGATION_PATTERNS = [
        r"\bnever\b", r"\bdid not\b", r"\bdidn't\b", r"\bnot\b",
        r"\bno\b", r"\bnone\b", r"\bwould not\b", r"\bwouldn't\b",
    ]
    EXCLUSIVE_WORDS = {"but", "except", "without", "exclude", "other than"}
    FILLER_PHRASES = {
        "to be honest", "honestly", "believe me", "i swear",
        "i promise", "trust me", "as a matter of fact",
    }  # truth-claim phrases — ironically common in deceptive speech

    async def analyze(self, audio_path: str, transcript: Optional[str] = None) -> NLPResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_sync, transcript)

    def _analyze_sync(self, transcript: Optional[str]) -> NLPResult:
        if not transcript or len(transcript.strip()) < 20:
            return self._fallback_result("No transcript available")

        text = transcript.lower()
        words = re.findall(r"\b\w+\b", text)
        sentences = re.split(r"[.!?]+", transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        word_count = len(words)

        if word_count == 0:
            return self._fallback_result("Empty transcript")

        # 1. Distancing language score
        first_person_count = sum(1 for w in words if w in self.FIRST_PERSON)
        distancing_count = sum(text.count(d) for d in self.DISTANCING)
        total_ref = first_person_count + distancing_count + 1
        distancing_score = min(1.0, distancing_count / total_ref)

        # 2. Spontaneous negation rate (per 100 words)
        negation_count = sum(
            len(re.findall(p, text)) for p in self.NEGATION_PATTERNS
        )
        negation_rate = min(1.0, (negation_count / word_count) * 10)

        # 3. Over-explanation: long sentences relative to mean
        if sentences:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            over_explanation_score = min(1.0, max(0.0, (mean_len - 15) / 20.0))
        else:
            over_explanation_score = 0.0

        # 4. Truth-claim phrase rate (paradoxically indicates deception)
        truth_claim_count = sum(text.count(p) for p in self.FILLER_PHRASES)
        truth_claim_rate = min(1.0, truth_claim_count / (word_count / 50 + 1))

        # 5. Exclusive word usage — honest accounts use more of these
        exclusive_count = sum(1 for w in words if w in self.EXCLUSIVE_WORDS)
        exclusive_ratio = exclusive_count / word_count
        cognitive_load_score = max(0.0, min(1.0, 0.5 - exclusive_ratio * 20))

        # 6. Contradiction detection (simple heuristic)
        contradiction_count = self._detect_contradictions(sentences)

        # Aggregate
        deception_score = (
            distancing_score * 0.25
            + negation_rate * 0.20
            + over_explanation_score * 0.15
            + truth_claim_rate * 0.15
            + cognitive_load_score * 0.15
            + min(1.0, contradiction_count / 3.0) * 0.10
        )

        flags = []
        if distancing_score > 0.5:
            flags.append("High use of distancing language (avoids 'I')")
        if negation_rate > 0.4:
            flags.append("Elevated spontaneous negation rate")
        if over_explanation_score > 0.6:
            flags.append("Over-explanation pattern detected")
        if truth_claim_count > 2:
            flags.append(f"{truth_claim_count} unsolicited truth claims ('believe me', etc.)")
        if contradiction_count > 0:
            flags.append(f"{contradiction_count} potential contradictions detected")

        return NLPResult(
            deception_score=round(deception_score, 3),
            confidence=0.65,
            distancing_score=round(distancing_score, 3),
            negation_rate=round(negation_rate, 3),
            over_explanation_score=round(over_explanation_score, 3),
            contradiction_count=contradiction_count,
            cognitive_load_score=round(cognitive_load_score, 3),
            flags=flags,
        )

    def _detect_contradictions(self, sentences: List[str]) -> int:
        """
        Simple contradiction heuristic: look for sentences that
        assert the opposite of a previous sentence.
        """
        count = 0
        affirmed = set()
        for sent in sentences:
            words = set(re.findall(r"\b\w+\b", sent.lower()))
            if any(neg in sent.lower() for neg in ["not", "never", "didn't", "don't"]):
                key_words = words - {"not", "never", "i", "the", "a", "is", "was"}
                overlap = key_words & affirmed
                if len(overlap) > 1:
                    count += 1
            else:
                affirmed.update(words)
        return count

    def _fallback_result(self, reason: str) -> NLPResult:
        return NLPResult(
            deception_score=0.0, confidence=0.0,
            distancing_score=0.0, negation_rate=0.0,
            over_explanation_score=0.0, contradiction_count=0,
            cognitive_load_score=0.0, flags=[reason],
        )
