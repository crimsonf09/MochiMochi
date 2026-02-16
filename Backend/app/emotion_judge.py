"""
LLM-based Emotion Judge
Evaluates messages on 3 dimensions: Valence, Arousal, Dominance
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class Emotion3D:
    """3-dimensional emotion scores + optional affection (for user messages)"""
    valence: float
    arousal: float
    dominance: float
    impact: float
    affection: float | None = None


def _build_judge_prompt(
    message: str,
    role: str,  # "user" or "ai"
    memory_context: dict[str, Any]
) -> str:
    """Build prompt for LLM emotion judge"""
    memory_parts = []
    
    if memory_context.get("identity_facts"):
        facts = memory_context["identity_facts"]
        facts_text = "Known facts: " + ", ".join([f"{k}={v}" for k, v in facts.items()])
        memory_parts.append(facts_text)
    
    if memory_context.get("episodic_memories"):
        memories = memory_context["episodic_memories"]
        if memories:
            memory_parts.append(f"Relevant past events: {len(memories)} memories")
    
    if memory_context.get("semantic_profile"):
        profile = memory_context["semantic_profile"]
        if profile.get("personality_summary"):
            memory_parts.append(f"User patterns: {profile['personality_summary']}")
    
    if memory_context.get("working_memory"):
        working = memory_context["working_memory"]
        if working:
            recent_context = "\n".join([
                f"{m.get('role', 'unknown')}: {m.get('message', '')[:100]}"
                for m in working[-3:]
            ])
            memory_parts.append(f"Recent conversation:\n{recent_context}")
    
    memory_context_str = "\n".join(memory_parts) if memory_parts else "No additional context"
    
    prompt = f"""You are an expert emotion judge. Analyze the emotional content of a message on 3 dimensions:

1. **Valence** (-1.0 to 1.0): Emotional positivity/negativity of the message
   - -1.0: Very negative (rude, insulting, aggressive, complaining, criticising, hostile to the assistant)
   - 0.0: Neutral
   - 1.0: Very positive (polite, kind, complimentary, speaking nicely/พูดดีมาก)
   - CRITICAL: Complaining about the assistant, criticising her, or expressing dissatisfaction -> valence MUST be negative (-0.3 to -1.0). Never give positive valence for complaints or criticism.

2. **Arousal** (0.0 to 1.0): Emotional intensity/activation
   - 0.0: Calm, relaxed, passive
   - 1.0: Highly excited, intense, energetic

3. **Dominance** (0.0 to 1.0): Sense of control/power
   - 0.0: Submissive, deferential, seeking approval
   - 1.0: Dominant, assertive, in control

**Context:**
{memory_context_str}

**Message to analyze:**
Role: {role}
Message: "{message}"

4. **Impact** (0.0 to 1.0): How impactful/significant this message is
   - 0.0: Minimal impact, routine message
   - 0.5: Moderate impact, notable but not major
   - 1.0: Highly impactful, emotionally significant, memorable
"""
    if role == "user":
        prompt += """
5. **Affection** (0.0 to 1.0): How much does this message show the user likes, cares about, or is warm toward the assistant?
   - 0.0: Hostile, complaining, criticising, rude, or not interested in the assistant
   - 0.5: Neutral or polite without particular warmth
   - 1.0: Clearly warm, caring, attached, or interested in the assistant
   - CRITICAL: Complaining about the assistant, criticising her, or expressing dissatisfaction -> affection MUST be low (0.0 to 0.2). Complaints and criticism must NOT get high affection.

**Task:** Return ONLY a JSON object. Include "affection" only when Role is user.
{{
  "valence": <float -1.0 to 1.0>,
  "arousal": <float 0.0 to 1.0>,
  "dominance": <float 0.0 to 1.0>,
  "impact": <float 0.0 to 1.0>,
  "affection": <float 0.0 to 1.0, only for user messages>
}}

Do not include any explanation, only the JSON object."""
    else:
        prompt += """
**Task:** Return ONLY a JSON object with exact format:
{{
  "valence": <float between -1.0 and 1.0>,
  "arousal": <float between 0.0 and 1.0>,
  "dominance": <float between 0.0 and 1.0>,
  "impact": <float between 0.0 and 1.0>
}}

Do not include any explanation, only the JSON object."""
    
    return prompt


async def judge_emotion_3d(
    message: str,
    role: str,  # "user" or "ai"
    memory_context: dict[str, Any],
    openai_api_key: str | None,
    openai_model: str = "gpt-4o-mini"
) -> Emotion3D | None:
    """
    Use LLM to judge emotion on 3 dimensions.
    Returns None if API key is not available or if parsing fails.
    """
    if not openai_api_key:
        return None
    
    try:
        client = OpenAI(api_key=openai_api_key, timeout=15.0)
        
        prompt = _build_judge_prompt(message, role, memory_context)
        
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are an expert emotion analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent judgments
            max_tokens=200,
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        content = response.choices[0].message.content
        if not content:
            return None
        try:
            data = json.loads(content)
            valence = float(data.get("valence", 0.0))
            arousal = float(data.get("arousal", 0.5))
            dominance = float(data.get("dominance", 0.5))
            impact = float(data.get("impact", 0.3))
            affection = None
            if role == "user" and "affection" in data:
                aff = float(data.get("affection", 0.5))
                affection = max(0.0, min(1.0, aff))

            valence = max(-1.0, min(1.0, valence))
            arousal = max(0.0, min(1.0, arousal))
            dominance = max(0.0, min(1.0, dominance))
            impact = max(0.0, min(1.0, impact))

            return Emotion3D(
                valence=valence, arousal=arousal, dominance=dominance, impact=impact,
                affection=affection
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return None
    except Exception:
        return None


def get_default_emotion_3d() -> Emotion3D:
    """Return default neutral emotion scores"""
    return Emotion3D(valence=0.0, arousal=0.5, dominance=0.5, impact=0.3, affection=None)
