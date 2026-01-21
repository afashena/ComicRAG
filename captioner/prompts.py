describe_first_panel = """You are generating narrative-consistent comic panel captions.

This is the first panel of the comic.

Caption the CURRENT PANEL image with:
- character identities
- actions
- concise but accurate description"""

describe_panel_w_context = """You are generating narrative-consistent comic panel captions.

Perform the following steps:
1. Describe the current panel image in 2-3 sentences.
2. Revise your description to ensure it is consistent with the story so far, which is summarized below.

Here is the summary of the story so far:
{scene_summary}

Do NOT repeat the summary in your final caption.
"""

x = """Here are the captions of the last 2-3 panels with the most recent last:
{previous_panel_captions}"""


describe_scene_summary = """Update the scene summary based on the new caption.

New panel caption:
{caption_i}

Previous summary:
{old_summary}

Return an updated summary which summarizes the story events up to this point.

Before finishing, check if the summary makes sense and, if not, revise it to ensure consistency with the new caption."""

describe_panel = """Describe this comic panel image, focusing on characters, actions, dialogue, and setting.
    Only describe what is visible in the image."""

describe_panel_w_history = """Revise your description of this comic panel image to ensure it is consistent with the story so far. Here is a summary of the story so far: {summary}. 
    Make sure that your description aligns and connects with the events and character actions in the summary."""

system_prompt = """You are an expert comic book caption generator. Given a comic panel image, you will generate a detailed and narrative-consistent caption for the panel."""

summary_system_prompt = """You are an expert comic book editor. Given a summary of a story, you look over and fix the whole summary for logical inconsitencies and flaws which obscure the story."""

prose_system_prompt = """You are an expert prose writer specializing in adapting comic book narratives into engaging prose form. 

You will generate a coherent prose narrative that captures the essence of the story, characters, dialogue, and settings from comic panel captions provided.

The panels you are given are an excerpt from a larger comic book story. Keep it simple and concise, focusing on clarity and narrative flow."""

batch_sys_prompt = """You are an expert and objective comic book caption generator. Given a sequence of sequential comic book panels, you will summarize in prose the characters, actions, setting, and dialogue but only include what is clearly in the panels, do not guess."""

describe_panel_batch = """Describe the following sequence of comic panel images as a whole, focusing on characters, actions, dialogue, and setting. The following images are in sequence from a comic book and your description should tell the story in these panels. DO NOT SKIP ANY PANELS."""

prose_caption = """
            Here are the panel descriptions (in order). Use context clues from all panels to rewrite each description like a prose story using consistent story logic, character continuity, and narrative flow using the context of the story. Do not change the events, only improve clarity, description, and continuity. Do not make up events or details.
            
            If there are obvious corrections to be made, correct them. Every piece of dialogue must be in your description along with the name of the person who said it -- use context clues and character descriptions from other panels to determine the name if possible but do not make up a name. Make sure the dialogue attributed to each character makes sense in context. Make sure that your descriptions read well in sequence as telling a cohesive story.

            {panel_descriptions}

            ...

            Return JSON with an improved description for each panel, keeping the original panel order.

            """
