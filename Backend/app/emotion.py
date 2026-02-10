from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionUpdate:
    delta: int
    new_score: int
    label: str


KIND_TERMS = {
    "thank",
    "thanks",
    "ty",
    "please",
    "sorry",
    "apologize",
    "appreciate",
    "great",
    "good",
    "awesome",
    "nice",
    "love",
    "cute",
    "kind",
    "helpful",
}

RUDE_TERMS = {
    "stupid",
    "idiot",
    "dumb",
    "shut",
    "hate",
    "annoying",
    "trash",
    "ugly",
    "worst",
    "loser",
    "moron",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z']+", (text or "").lower()))


def sentiment_delta(user_text: str) -> int:
    """
    Deterministic, rule-based sentiment delta.
    - Kind/polite terms -> +1
    - Rude/insulting terms -> -1
    If both appear, cancel to 0.
    """
    tokens = _tokenize(user_text)
    kind = len(tokens & KIND_TERMS) > 0
    rude = len(tokens & RUDE_TERMS) > 0
    if kind and not rude:
        return 1
    if rude and not kind:
        return -1
    return 0


def emotion_label_for_score(score: int) -> str:
    if score <= -3:
        return "Hostile Tsundere"
    if -2 <= score <= 1:
        return "Cold / Defensive"
    if 2 <= score <= 4:
        return "Soft Tsundere"
    return "Dere Mode"


def apply_emotion_update(prev_score: int, user_text: str) -> EmotionUpdate:
    d = sentiment_delta(user_text)
    new_score = prev_score + d
    return EmotionUpdate(delta=d, new_score=new_score, label=emotion_label_for_score(new_score))

