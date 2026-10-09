"""Weather-only prompts for Cosmos3 transfer (gen/modal_cosmos3.py).

Unlike gen/prompts.py these never mention sun or shadows: with edge control the seed's clear-day look
wins unless the prompt pushes the whole frame (sky, contrast, color) toward the condition.
"""

LAYOUT = {
    "highway": "A locked-off elevated traffic camera looks along a multi-lane interstate. The same cars and trucks hold their lanes.",
    "street": "A locked-off traffic camera looks down a city street intersection. The same cars, buses and pedestrians keep their places.",
}

WEATHER = {
    "fog": ("dense fog. Thick grey fog fills the air and drops visibility to about 50 m: distant lanes, trees and buildings "
            "fade into a flat white-grey wall, the sky is uniformly overcast with no sun and no shadows, colors are washed "
            "out and low contrast, and headlights glow as soft halos."),
    "rain": ("heavy rain. A dark overcast sky with no sun and no shadows, sheets of rain streak through the frame, the road "
             "is soaked and mirror-wet with reflections of headlights and taillights, tire spray hangs behind vehicles, "
             "and the distance is grey and hazy."),
    "snow": ("a heavy snowstorm. Thick falling snow veils the scene, the sky is flat white-grey with no sun and no shadows, "
             "snow covers the shoulders, medians and roofs, the road is slushy white-grey, and distant objects fade into "
             "the snowfall."),
    "night-rain": ("heavy rain at night. It is dark, lit only by streetlights and vehicle headlights, rain streaks through "
                   "the light, the wet road mirrors long glowing reflections of headlights and taillights, and everything "
                   "beyond the lamps falls into darkness."),
}


def prompt(scene: str, weather: str) -> str:
    head, tail = LAYOUT[scene].split(". ", 1)
    return f"{head} in {WEATHER[weather]} {tail}"
