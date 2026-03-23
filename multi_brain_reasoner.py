"""
MultiBrainReasoner — sends TruthLens signal scores to THREE AI models
in parallel and reconciles their verdicts for maximum accuracy.

Models:
  - Claude Sonnet  (Anthropic)  — nuanced, cautious, structured reasoning
  - GPT-4o         (OpenAI)     — broad training, different perspective
  - DeepSeek R1    (DeepSeek)   — strong step-by-step analytical reasoning

Flow:
  1. All 4 models receive identical signal data simultaneously (asyncio.gather)
  2. Each returns: score (0-1), explanation, key_observations, confidence
  3. Reconciler (Claude) weighs all 3 verdicts:
     - All agree     → high confidence final verdict
     - 2 of 3 agree  → majority verdict + note the dissent
     - All disagree  → flag inconclusive, show all 3 perspectives
  4. Final output includes consensus score + per-model breakdown for the UI
"""

import asyncio
import json
import anthropic
from openai import AsyncOpenAI
from dataclasses import dataclass, field
from typing import List, Optional, Any
from config import settings


@dataclass
class ModelVerdict:
    model_name: str
    score: float
    explanation: str
    key_observations: List[str]
    confidence: str            # "high" | "medium" | "low"
    error: Optional[str] = None


@dataclass
class MultiBrainOutput:
    final_score: float
    consensus: str             # "full" | "majority" | "split"
    explanation: str
    key_observations: List[str]
    confidence_rationale: str
    claude_verdict: Optional[ModelVerdict] = None
    gpt4o_verdict: Optional[ModelVerdict] = None
    deepseek_verdict: Optional[ModelVerdict] = None
    minimax_verdict: Optional[ModelVerdict] = None
    agreeing_models: List[str] = field(default_factory=list)
    dissenting_models: List[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are a forensic AI analyst for TruthLens, an app that detects
deception and AI-generated video. You receive structured signal data from multiple
detection models and must reason holistically across all of them.

Rules:
1. Reason across ALL signals — not just the highest number
2. Cross-signal confirmation (facial stress AND voice pitch up) = stronger evidence
3. Contradicting signals = lower confidence — explain why they conflict
4. Never say the person is "lying" — say signals are elevated, patterns are consistent with stress
5. Be measured, fair, and probabilistic — this is not proof

Respond ONLY in this exact JSON format, no extra text:
{
  "score": 0.0,
  "explanation": "2-3 sentence plain English summary",
  "key_observations": ["observation 1", "observation 2", "observation 3"],
  "confidence": "high|medium|low"
}"""


class MultiBrainReasoner:
    def __init__(self):
        self.claude_client = anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY
        )
        self.openai_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
        )
        # DeepSeek uses OpenAI-compatible REST API
        self.deepseek_client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
        self.minimax = AsyncOpenAI(
            api_key=settings.MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
        )

    async def reason(
        self,
        session_id: str,
        is_ai_generated: bool,
        deepfake_breakdown: Any,
        signal_breakdown: Any = None,
        facial_detail: Any = None,
        pose_detail: Any = None,
        voice_detail: Any = None,
        nlp_detail: Any = None,
        duration: float = 0.0,
    ) -> MultiBrainOutput:

        prompt = (
            self._build_deepfake_prompt(deepfake_breakdown, duration)
            if is_ai_generated
            else self._build_deception_prompt(
                deepfake_breakdown, signal_breakdown,
                facial_detail, pose_detail, voice_detail, nlp_detail, duration
            )
        )

        # All 4 models run simultaneously
        claude_v, gpt4o_v, deepseek_v, minimax_v = await asyncio.gather(
            self._ask_claude(prompt),
            self._ask_gpt4o(prompt),
            self._ask_deepseek(prompt),
            self._ask_minimax(prompt),
        )

        return await self._reconcile(
            claude_v, gpt4o_v, deepseek_v, minimax_v, is_ai_generated
        )

    # ── Model callers ────────────────────────────────────────────────────────

    async def _ask_claude(self, prompt: str) -> ModelVerdict:
        try:
            msg = await self.claude_client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=800,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_verdict("Claude Sonnet", msg.content[0].text)
        except Exception as e:
            return ModelVerdict(
                model_name="Claude Sonnet", score=0.5,
                explanation="Claude unavailable.", key_observations=[],
                confidence="low", error=str(e),
            )

    async def _ask_gpt4o(self, prompt: str) -> ModelVerdict:
        try:
            resp = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=800,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return self._parse_verdict("GPT-4o", resp.choices[0].message.content)
        except Exception as e:
            return ModelVerdict(
                model_name="GPT-4o", score=0.5,
                explanation="GPT-4o unavailable.", key_observations=[],
                confidence="low", error=str(e),
            )

    async def _ask_deepseek(self, prompt: str) -> ModelVerdict:
        try:
            resp = await self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                max_tokens=800,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return self._parse_verdict("DeepSeek R1", resp.choices[0].message.content)
        except Exception as e:
            return ModelVerdict(
                model_name="DeepSeek R1", score=0.5,
                explanation="DeepSeek unavailable.", key_observations=[],
                confidence="low", error=str(e),
            )

    async def _ask_minimax(self, prompt: str) -> ModelVerdict:
        try:
            resp = await self.minimax.chat.completions.create(
                model="minimax-m2.7",
                max_tokens=800,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return self._parse_verdict("MiniMax M2.7", resp.choices[0].message.content)
        except Exception as e:
            return ModelVerdict(
                model_name="MiniMax M2.7", score=0.5,
                explanation="MiniMax unavailable.", key_observations=[],
                confidence="low", error=str(e),
            )

    # ── Reconciler ───────────────────────────────────────────────────────────

    async def _reconcile(
        self,
        claude_v: ModelVerdict,
        gpt4o_v: ModelVerdict,
        deepseek_v: ModelVerdict,
        minimax_v: ModelVerdict,
        is_ai_generated: bool,
    ) -> MultiBrainOutput:

        verdicts = [claude_v, gpt4o_v, deepseek_v, minimax_v]
        valid = [v for v in verdicts if not v.error]
        scores = [v.score for v in valid]

        if not scores:
            return MultiBrainOutput(
                final_score=0.5, consensus="split",
                explanation="All models unavailable — result inconclusive.",
                key_observations=[], confidence_rationale="No models responded.",
            )

        # Models "agree" if scores are within 0.20 of each other
        def agree(a: ModelVerdict, b: ModelVerdict) -> bool:
            return abs(a.score - b.score) <= 0.20

                # 4-model consensus
        def count_agreements(v):
            return sum(1 for o in valid if o.model_name != v.model_name and agree(v, o))

        if len(valid) >= 3:
            majority_models = [v for v in valid if count_agreements(v) >= 2]
            minority_models = [v for v in valid if count_agreements(v) < 2]
            all_agree = len(minority_models) == 0
        elif len(valid) == 2:
            all_agree = agree(valid[0], valid[1])
            majority_models = valid if all_agree else [valid[0]]
            minority_models = [] if all_agree else [valid[1]]
        else:
            all_agree = True
            majority_models = valid
            minority_models = []

        if all_agree:
            consensus = "full"
            agreeing = [v.model_name for v in valid]
            dissenting = []
            final_score = sum(scores) / len(scores)
        elif len(majority_models) >= 2:
            consensus = "majority"
            agreeing = [v.model_name for v in majority_models]
            dissenting = [v.model_name for v in minority_models]
            final_score = sum(v.score for v in majority_models) / len(majority_models)
        else:
            consensus = "split"
            agreeing, dissenting = [], [v.model_name for v in valid]
            final_score = sum(scores) / len(scores)

        # Claude writes the final reconciled user-facing explanation
        explanation, observations, confidence_rationale = await self._write_reconciled_summary(
            claude_v, gpt4o_v, deepseek_v, minimax_v, consensus, final_score, is_ai_generated
        )

        return MultiBrainOutput(
            final_score=round(final_score, 3),
            consensus=consensus,
            explanation=explanation,
            key_observations=observations,
            confidence_rationale=confidence_rationale,
            claude_verdict=claude_v,
            gpt4o_verdict=gpt4o_v,
            deepseek_verdict=deepseek_v,
            minimax_verdict=minimax_v,
            agreeing_models=agreeing,
            dissenting_models=dissenting,
        )

    async def _write_reconciled_summary(
        self,
        claude_v: ModelVerdict,
        gpt4o_v: ModelVerdict,
        deepseek_v: ModelVerdict,
        minimax_v: ModelVerdict,
        consensus: str,
        final_score: float,
        is_ai_generated: bool,
    ) -> tuple:
        analysis_type = "AI video detection" if is_ai_generated else "deception detection"
        consensus_note = {
            "full":     "All 3 models broadly agree.",
            "majority": "2 of 3 models agree — briefly note the dissenting view.",
            "split":    "Models significantly disagree — explain this uncertainty fairly to the user.",
        }[consensus]

        prompt = f"""Reconcile these 3 AI verdicts for {analysis_type}:

CONSENSUS: {consensus.upper()} | FINAL SCORE: {final_score:.3f}

CLAUDE SONNET (score {claude_v.score:.3f}, {claude_v.confidence}):
"{claude_v.explanation}"
Observations: {claude_v.key_observations}

GPT-4O (score {gpt4o_v.score:.3f}, {gpt4o_v.confidence}):
"{gpt4o_v.explanation}"
Observations: {gpt4o_v.key_observations}

DEEPSEEK R1 (score {deepseek_v.score:.3f}, {deepseek_v.confidence}):
"{deepseek_v.explanation}"
Observations: {deepseek_v.key_observations}

MINIMAX M2.7 (score {minimax_v.score:.3f}, {minimax_v.confidence}):
"{minimax_v.explanation}"
Observations: {minimax_v.key_observations}

{consensus_note}

Respond ONLY in JSON:
{{
  "explanation": "2-3 sentences for the end user",
  "key_observations": ["obs1", "obs2", "obs3"],
  "confidence_rationale": "1 sentence"
}}"""

        try:
            msg = await self.claude_client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=500,
                system="You write final reconciled verdicts for TruthLens. Be fair, clear and non-accusatory. JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            data = self._parse_json(msg.content[0].text)
            return (
                data.get("explanation", "Analysis complete."),
                data.get("key_observations", []),
                data.get("confidence_rationale", ""),
            )
        except Exception:
            return (
                f"{consensus.capitalize()} verdict across models. Score: {final_score:.2f}.",
                [],
                f"Based on {3 - sum(1 for v in [claude_v, gpt4o_v, deepseek_v] if v.error)}/3 models.",
            )

    # ── Prompt builders ──────────────────────────────────────────────────────

    def _build_deepfake_prompt(self, deepfake: Any, duration: float) -> str:
        flags = getattr(deepfake, "_flags", [])
        return f"""TASK: AI-Generated Video Detection

Duration: {duration:.1f}s

FORENSIC ZONE SCORES (0=natural, 1=AI-generated):
- Face / biological signals:  {deepfake.biological_signals_score:.3f}
- Visual artifacts:           {deepfake.visual_artifacts_score:.3f}
- Frequency / GAN artifacts:  {deepfake.frequency_artifacts_score:.3f}
- Audio (TTS/clone/reverb):   {deepfake.av_sync_score:.3f}
- Metadata / EXIF:            {deepfake.metadata_score:.3f}
- Temporal consistency:       {deepfake.temporal_consistency_score:.3f}
- ENSEMBLE SCORE:             {deepfake.ensemble_score:.3f}

FLAGS: {flags if flags else 'none'}

Reason across all zones. Which are most decisive? Return JSON verdict."""

    def _build_deception_prompt(
        self, deepfake: Any, signals: Any, facial: Any,
        pose: Any, voice: Any, nlp: Any, duration: float
    ) -> str:
        transcript = (
            voice.transcript[:400]
            if voice and getattr(voice, "transcript", None)
            else "Not available"
        )
        return f"""TASK: Deception Detection (video is real — deepfake score {deepfake.ensemble_score:.2f})

Duration: {duration:.1f}s

FACIAL:
- Stress:           {getattr(signals, 'facial_au_score', 'N/A')}
- Gaze aversion:    {getattr(facial, 'gaze_aversion_rate', 'N/A')}
- Smile asymmetry:  {getattr(facial, 'smile_asymmetry', 'N/A')}
- Micro-expressions:{getattr(facial, 'micro_expression_count', 'N/A')}
- Flags: {getattr(facial, 'flags', [])}

BODY:
- Score:            {getattr(signals, 'body_language_score', 'N/A')}
- Self-touch:       {getattr(pose, 'self_touch_rate', 'N/A')}
- Posture shifts:   {getattr(pose, 'posture_shift_count', 'N/A')}
- Flags: {getattr(pose, 'flags', [])}

VOICE:
- Score:            {getattr(signals, 'voice_prosody_score', 'N/A')}
- Pitch:            {getattr(voice, 'pitch_mean', 'N/A')} Hz
- Pauses/min:       {getattr(voice, 'pause_rate', 'N/A')}
- Fillers/min:      {getattr(voice, 'filler_word_rate', 'N/A')}
- Flags: {getattr(voice, 'flags', [])}

LANGUAGE:
- Score:            {getattr(signals, 'nlp_score', 'N/A')}
- Distancing:       {getattr(nlp, 'distancing_score', 'N/A')}
- Negations:        {getattr(nlp, 'negation_rate', 'N/A')}
- Contradictions:   {getattr(nlp, 'contradiction_count', 'N/A')}
- Flags: {getattr(nlp, 'flags', [])}

TRANSCRIPT: {transcript}

Reason across all channels. Note convergences and contradictions. Return JSON verdict."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_verdict(self, model_name: str, raw: str) -> ModelVerdict:
        try:
            data = self._parse_json(raw)
            return ModelVerdict(
                model_name=model_name,
                score=max(0.0, min(1.0, float(data.get("score", 0.5)))),
                explanation=data.get("explanation", ""),
                key_observations=data.get("key_observations", []),
                confidence=data.get("confidence", "medium"),
            )
        except Exception as e:
            return ModelVerdict(
                model_name=model_name, score=0.5,
                explanation=raw[:300], key_observations=[],
                confidence="low", error=str(e),
            )

    def _parse_json(self, raw: str) -> dict:
        clean = raw.strip()
        for fence in ["```json", "```"]:
            if clean.startswith(fence):
                clean = clean[len(fence):]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip())
