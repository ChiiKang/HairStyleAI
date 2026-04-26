"""
Hairstyle prompts for AI generation.

Each hairstyle has:
- id: unique identifier
- label: display name shown in the UI
- edit_prompt: used with image-editing models (modifies the uploaded selfie)
- generate_prompt: used with text-to-image models (generates a new image)

Edit these prompts to improve generation quality.
"""

HAIRSTYLES = [
    {
        "id": "protective-braids",
        "label": "Protective Braids",
        "edit_prompt": (
            "Transform this person's hairstyle into long, neat protective box braids. "
            "Keep the person's face, skin tone, and features exactly the same. "
            "The braids should be medium-sized, well-defined, and fall past the shoulders. "
            "Maintain the original photo's lighting, background, and composition."
        ),
        "generate_prompt": (
            "A photorealistic portrait of a beautiful Black woman with long, neat protective "
            "box braids falling past her shoulders. The braids are medium-sized and well-defined. "
            "Natural skin, soft studio lighting, clean background, high detail, professional photo."
        ),
    },
    {
        "id": "natural-twist-out",
        "label": "Natural Twist-Out",
        "edit_prompt": (
            "Transform this person's hairstyle into a voluminous natural twist-out. "
            "Keep the person's face, skin tone, and features exactly the same. "
            "The twist-out should show defined, bouncy curls with natural volume and texture. "
            "Maintain the original photo's lighting, background, and composition."
        ),
        "generate_prompt": (
            "A photorealistic portrait of a beautiful Black woman with a voluminous natural "
            "twist-out hairstyle. Defined, bouncy curls with beautiful natural texture and volume. "
            "Natural skin, soft studio lighting, clean background, high detail, professional photo."
        ),
    },
    {
        "id": "silk-press",
        "label": "Silk Press",
        "edit_prompt": (
            "Transform this person's hairstyle into a sleek, smooth silk press. "
            "Keep the person's face, skin tone, and features exactly the same. "
            "The hair should be straight, glossy, and flowing with a healthy shine. "
            "Maintain the original photo's lighting, background, and composition."
        ),
        "generate_prompt": (
            "A photorealistic portrait of a beautiful Black woman with a sleek, smooth silk press "
            "hairstyle. The hair is straight, glossy, and flowing with a healthy shine. "
            "Natural skin, soft studio lighting, clean background, high detail, professional photo."
        ),
    },
    {
        "id": "bantu-knots",
        "label": "Bantu Knots",
        "edit_prompt": (
            "Transform this person's hairstyle into neat, symmetrical Bantu knots. "
            "Keep the person's face, skin tone, and features exactly the same. "
            "The Bantu knots should be evenly spaced, tightly coiled, and well-formed. "
            "Maintain the original photo's lighting, background, and composition."
        ),
        "generate_prompt": (
            "A photorealistic portrait of a beautiful Black woman with neat, symmetrical Bantu "
            "knots hairstyle. The knots are evenly spaced, tightly coiled, and well-formed. "
            "Natural skin, soft studio lighting, clean background, high detail, professional photo."
        ),
    },
]


def get_hairstyle_prompts(use_edit: bool = True) -> list[dict]:
    """Return hairstyle configs with the appropriate prompt field selected."""
    key = "edit_prompt" if use_edit else "generate_prompt"
    return [
        {"id": h["id"], "label": h["label"], "prompt": h[key]}
        for h in HAIRSTYLES
    ]
