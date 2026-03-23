"""
Claude fusion engine.
Receives all signal outputs, reasons holistically,
and returns a deception score + explanation.
"""
import json
import anthropic
from dataclasses import asdict
from typing import Optional
import structlog

from config import settings
from facial import FacialResult
from voice import VoiceResult
from nlp import NLPResult
from deepfake import DeepfakeResult

log = structlog.get_logger()

client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are TruthLens, an expert deception analysis AI.
You receive structured signal data from multiple detectors analyzing a video.
Your job is to reason across ALL signals holistically and produce a deception assessment.

Rules:
- Never make definitive claims — use probabilistic language
- Weight signals by their reliability and context
- Flag contradictions between signals as especially meaningful
- Consider that stress does not equal deception — note confounding factors
- If the video is AI-generated, deception analysis is invalid and you must say so

Respond ONLY with valid JSON in this exact format:
{
  "deception_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "verdict": "low_risk" | "moderate_risk" | "high_risk" | "ai_generated" | "inconclusive",
  "key_indicators": ["string", ...],
  "reasoning": "2-4 sentence plain English explanation",
  "signal_weights": {
    "facial": 0.0-1.0,
    "voice": 0.0-1.0,
    "nlp": 0.0-1.0,
    "body": 0.0-1.0
  },
  "confounding_factors": ["string", ...]
}"""


def build_analysis_prompt(
    facial: Optional[FacialResult],
    voice: Optional[VoiceResult],
    nlp: Optional[NLPResult],
    deepfake: Optional[DeepfakeResult],
    baseline: Optional[dict] = None,
) -> str:
    sections = []

    if deepfake:
        sections.append(f"""## AI/Deepfake Detection
Ensemble score: {deepfake.ensemble_score:.3f} (threshold: {settings.DEEPFAKE_THRESHOLD})
Is AI generated: {deepfake.is_ai_generated}
- Visual artifacts: {deepfake.visual_artifact_score:.3f}
- Temporal flicker: {deepfake.temporal_flicker_score:.3f}
- rPPG absence (no heartbeat): {deepfake.rppg_absence_score:.3f}
- Frequency artifacts: {deepfake.frequency_artifact_score:.3f}
- Metadata anomalies: {deepfake.metadata_anomaly_score:.3f}""")

        if deepfake.is_ai_generated:
            sections.append("⚠️ HIGH CONFIDENCE AI-GENERATED VIDEO — deception analysis below is for reference only.")

    if facial:
        sections.append(f"""## Facial Signals
- Micro-expression stress score: {facial.micro_expression_score:.3f}
- Gaze aversion: {facial.gaze_aversion_score:.3f}
- Blink irregularity: {facial.blink_irregularity:.3f}
- Lip compression: {facial.lip_compression:.3f}
- Asymmetric smile: {facial.asymmetric_smile:.3f}
Key AUs: {json.dumps(facial.raw_aus, indent=2)}""")

    if voice:
        sections.append(f"""## Voice Prosody Signals
- Pitch elevation vs baseline: {voice.pitch_elevation:.3f}
- Speech rate change: {voice.speech_rate_change:.3f}
- Filler word rate: {voice.filler_word_rate:.3f}
- Pause before answer score: {voice.pause_before_answer:.3f}
- Voice tremor: {voice.voice_tremor:.3f}
- Duration: {voice.duration_seconds:.1f}s, Words: {voice.word_count}""")

    if nlp:
        sections.append(f"""## Linguistic / NLP Signals
- Distancing language: {nlp.distancing_language:.3f}
- Over-explanation: {nlp.over_explanation_score:.3f}
- Spontaneous negation: {nlp.spontaneous_negation_score:.3f}
- Contradiction count: {nlp.contradiction_count}
Transcript excerpt: "{nlp.transcript[:400]}..."
""")

    if baseline:
        sections.append(f"""## Baseline Comparison
Subject baseline available: {json.dumps(baseline, indent=2)}
(Deviations from baseline are more significant than absolute values)""")
    else:
        sections.append("## Baseline\nNo personal baseline available — using population norms.")

    return "\n\n".join(sections)


async def run_fusion(
    facial: Optional[FacialResult] = None,
    voice: Optional[VoiceResult] = None,
    nlp: Optional[NLPResult] = None,
    deepfake: Optional[DeepfakeResult] = None,
    baseline: Optional[dict] = None,
) -> dict:
    prompt = build_analysis_prompt(facial, voice, nlp, deepfake, baseline)

    log.info("Sending to Claude for fusion analysis")

    response = client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)
    log.info("Claude fusion complete", verdict=result.get("verdict"), score=result.get("deception_score"))
    return result
