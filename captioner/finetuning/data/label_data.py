from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class PanelType(str, Enum):
    establishing = "establishing"   # wide shot showing location
    action = "action"               # physical action happening
    reaction = "reaction"           # character reacting to something
    dialogue = "dialogue"           # conversation focused
    close_up = "close_up"           # face or object detail
    transition = "transition"       # panel connecting two scenes
    splash = "splash"               # full page or dramatic single image


class Emotion(str, Enum):
    neutral = "neutral"
    happy = "happy"
    angry = "angry"
    fearful = "fearful"
    sad = "sad"
    surprised = "surprised"
    determined = "determined"
    confused = "confused"


class CharacterAppearance(BaseModel):
    name: str = Field(
        description="Character name, or 'unknown_male_1' etc if unnamed."
    )
    emotion: Emotion
    action: str = Field(
        description="What this character is physically doing. One sentence."
    )


class ComicPanelCaption(BaseModel):
    # --- Structured fields (fast to fill, consistent, good for retrieval) ---
    panel_type: PanelType

    characters: List[CharacterAppearance] = Field(
        description="Every character visible in the panel."
    ) 

    dialogue: List[str] = Field(
        description=(
            "Every speech bubble and caption box verbatim, in reading order. "
            "Empty list if none."
        )
    ) 

    setting: str = Field(
        description=(
            "Location in 2-5 words. Be consistent: always 'rooftop' not "
            "'the roof' or 'top of building'."
        )
    ) 

    tags: List[str] = Field(
        description=(
            "3-8 controlled vocabulary keywords. Use the same tag every time "
            "for the same concept: 'fight', 'explosion', 'crowd', etc."
        )
    ) 

    # --- Prose anchor (captures nuance the structure misses) ---
    description: str = Field(
        description=(
            "2-4 sentences describing what is happening and why it matters "
            "narratively. Write this last, after filling the structured fields."
        )
    ) 