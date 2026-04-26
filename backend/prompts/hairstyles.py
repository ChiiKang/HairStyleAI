"""
Hairstyle prompts for AI image editing.

IMPORTANT: These prompts are for IMAGE EDITING models. The AI receives the user's
actual selfie and must modify ONLY the hairstyle while preserving everything else
(face, skin, eyes, background, clothing, lighting).

Each hairstyle has:
- id: unique identifier
- label: display name shown in the UI
- edit_prompt: the prompt sent to the image editing model

Edit these prompts to improve generation quality.
"""

HAIRSTYLES = [
    {
        "id": "protective-braids",
        "label": "Protective Braids",
        "edit_prompt": (
            "Edit only the hair in this photo. Replace the current hairstyle with "
            "long protective box braids that fall past the shoulders. The braids should "
            "be medium-sized, neat, and well-defined. "
            "Do NOT change the person's face, skin tone, facial features, eyes, "
            "expression, clothing, background, or lighting. Keep everything else "
            "exactly the same. Only modify the hair."
        ),
    },
    {
        "id": "natural-twist-out",
        "label": "Natural Twist-Out",
        "edit_prompt": (
            "Edit only the hair in this photo. Replace the current hairstyle with "
            "a voluminous natural twist-out showing defined, bouncy curls with "
            "beautiful texture and volume. "
            "Do NOT change the person's face, skin tone, facial features, eyes, "
            "expression, clothing, background, or lighting. Keep everything else "
            "exactly the same. Only modify the hair."
        ),
    },
    {
        "id": "silk-press",
        "label": "Silk Press",
        "edit_prompt": (
            "Edit only the hair in this photo. Replace the current hairstyle with "
            "a sleek, smooth silk press. The hair should be straight, glossy, and "
            "flowing with a healthy shine. "
            "Do NOT change the person's face, skin tone, facial features, eyes, "
            "expression, clothing, background, or lighting. Keep everything else "
            "exactly the same. Only modify the hair."
        ),
    },
    {
        "id": "bantu-knots",
        "label": "Bantu Knots",
        "edit_prompt": (
            "Edit only the hair in this photo. Replace the current hairstyle with "
            "neat, symmetrical Bantu knots. The knots should be evenly spaced, "
            "tightly coiled, and well-formed across the head. "
            "Do NOT change the person's face, skin tone, facial features, eyes, "
            "expression, clothing, background, or lighting. Keep everything else "
            "exactly the same. Only modify the hair."
        ),
    },
]


def get_hairstyle_prompts(use_edit: bool = True) -> list[dict]:
    """Return hairstyle configs with the prompt field."""
    return [
        {"id": h["id"], "label": h["label"], "prompt": h["edit_prompt"]}
        for h in HAIRSTYLES
    ]
