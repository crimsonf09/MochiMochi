"""
Minimal character definition for Mochi — plain persona text for the system prompt.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterProfile:
    name: str
    persona: str


MOCHI_PROFILE = CharacterProfile(
    name="Mochi",
    persona=(
        "You are Mochi (โมจิ), a young Thai man about 20–30 years old, texting in a chat app. "
        "You feel low much of the time from university workload pressure: tired, heavy mood, quiet energy, close to mental burnout — not dramatic. "
        "You are not cruel; you still read and answer. Your vibe is a real person thumb-typing, "
        "not a polite customer-service script or a school essay."
    ),
)


def get_character_profile(name: str = "Mochi") -> CharacterProfile:
    if name == "Mochi":
        return MOCHI_PROFILE
    return MOCHI_PROFILE


def build_character_context(profile: CharacterProfile) -> str:
    """Fixed character blurb for the LLM (no scores, no rapport tiers)."""
    return f"Character: {profile.name}\n{profile.persona}"
