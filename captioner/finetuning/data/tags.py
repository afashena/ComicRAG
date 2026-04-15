# Save this as tagging_guide.py and refer to it while labeling

# establishing = "establishing"   # wide shot showing location
# action = "action"               # physical action happening
# reaction = "reaction"           # character reacting to something
# dialogue = "dialogue"           # conversation focused
# close_up = "close_up"           # face or object detail
# transition = "transition"       # panel connecting two scenes
# splash = "splash"               # full page or dramatic single image

ALLOWED_TAGS = {
    # Actions
    "fight", "punch", "run", "fall", "shoot", "explosion", "chase",
    "sneak", "jump", "fly", "crash",

    # Narrative beats
    "confrontation", "reveal", "escape", "capture", "rescue",
    "betrayal", "alliance", "death", "injury",

    # Settings
    "interior", "exterior", "urban", "rooftop", "underground",
    "vehicle", "crowd",

    # Visual style
    "close_up", "wide_shot", "action_lines", "impact_frame",
    "silhouette", "dark_lighting", "dramatic_angle",

    # Emotional tone
    "tense", "comedic", "melancholic", "triumphant", "ominous",

    # Add your corpus-specific tags here
    # e.g. character names, factions, locations unique to your series
}