#!/usr/bin/env python3
"""Cosmos Transfer prompts for the condition grid.

Each prompt locks the seed's camera and layout and only changes weather,
lighting, and visibility. clear / day / light is the identity control.
"""

WEATHERS = ("clear", "fog", "rain", "snow")
TIMES = ("day", "dusk", "night")
INTENSITIES = ("light", "heavy")

# 4 x 3 x 2 = 24 arms. clear/day/light is the identity control.
CONDITIONS = [
    {"weather": w, "time": t, "intensity": i}
    for w in WEATHERS
    for t in TIMES
    for i in INTENSITIES
]

SCENES = ("highway", "warehouse")

# For a negative-prompt field, if the caller has one. Not pasted into prompt_for().
NEGATIVE = (
    "cartoon, illustration, anime, painting, extra vehicles, missing vehicles, "
    "extra people, changed lane markings, moved racks, new camera angle, "
    "fisheye, text, watermark, warped geometry"
)

_LAYOUT = {
    "highway": (
        "A locked-off elevated traffic camera looks along a multi-lane interstate, "
        "with the same lane markings, concrete barriers, cars, and trucks holding their lanes."
    ),
    "warehouse": (
        "A locked-off high surveillance camera looks down one warehouse aisle between tall pallet racks, "
        "with the same forklifts, pallets, and workers holding their places."
    ),
}

_LIGHT = {
    ("highway", "day", "light"): "It is bright midday with ordinary sun and short shadows on the pavement.",
    ("highway", "day", "heavy"): "It is harsh midday with a high hard sun and deep black shadows across the lanes.",
    ("highway", "dusk", "light"): "It is early dusk with a low warm sun and headlights just switching on.",
    ("highway", "dusk", "heavy"): "It is late dusk, a deep blue sky against a bright orange horizon, with headlights blooming.",
    ("highway", "night", "light"): "It is night, dark but readable, with streetlights and headlights on.",
    ("highway", "night", "heavy"): "It is a pitch-dark night, with pools of streetlight and headlights as the only source.",
    ("warehouse", "day", "light"): "Overhead aisle lights are on at an ordinary daytime level, with short soft shadows.",
    ("warehouse", "day", "heavy"): "Overhead aisle lights are harsh and bright, with deep shadows under the racks.",
    ("warehouse", "dusk", "light"): "The aisle is in early-dusk light, warm and low, with rack lamps just coming up.",
    ("warehouse", "dusk", "heavy"): "The aisle is in late-dusk light, deep blue with a warm spill, and rack lamps are blooming.",
    ("warehouse", "night", "light"): "The aisle is night-dark but readable, lit by rack lights and forklift headlights.",
    ("warehouse", "night", "heavy"): "The aisle is nearly black beyond pools of rack light and forklift headlights.",
}

# (scene, weather, intensity) -> one concrete sentence. Surfaces differ; layout does not.
_WEATHER = {
    ("highway", "clear", "light"): (
        "The air is clear with long visibility, the asphalt is dry, and contrast stays natural."
    ),
    ("highway", "clear", "heavy"): (
        "The air is crystal clear with no haze, contrast is hard, and the asphalt stays dry."
    ),
    ("highway", "fog", "light"): (
        "Thin fog softens the distance, visibility about 200 m, and far vehicles lose a little edge."
    ),
    ("highway", "fog", "heavy"): (
        "Dense fog drops visibility to about 50 m, headlights bloom into halos, and lane paint fades within a few car lengths."
    ),
    ("highway", "rain", "light"): (
        "Light rain draws fine streaks, and the damp asphalt carries soft reflections of the sky and lamps."
    ),
    ("highway", "rain", "heavy"): (
        "Heavy rain cuts visibility to about 40 m, wet asphalt mirrors headlights and taillights, and tire spray hangs in the lanes."
    ),
    ("highway", "snow", "light"): (
        "Light flurries drift through the frame, with a thin dusting on the shoulders and barriers and still-long visibility."
    ),
    ("highway", "snow", "heavy"): (
        "Heavy snow drops visibility to about 80 m, flakes veil the far lanes, and snow sits on barriers and vehicle roofs."
    ),
    ("warehouse", "clear", "light"): (
        "The air in the aisle is clear, the concrete floor is dry, and the full aisle stays visible."
    ),
    ("warehouse", "clear", "heavy"): (
        "The air is crystal clear with no haze, contrast is hard, and the concrete floor stays dry out to the far racks."
    ),
    ("warehouse", "fog", "light"): (
        "A light haze hangs in the aisle, visibility about 40 m, and the far racks soften."
    ),
    ("warehouse", "fog", "heavy"): (
        "Dense fog fills the aisle, visibility about 15 m, rack lights bloom, and the far racks disappear."
    ),
    ("warehouse", "rain", "light"): (
        "Light rain streaks across the lens, and the damp concrete floor shows soft reflections of the rack lights."
    ),
    ("warehouse", "rain", "heavy"): (
        "Heavy rain streaks across the lens, visibility about 20 m, and the soaked concrete mirrors the forklift lights."
    ),
    ("warehouse", "snow", "light"): (
        "Light snowflakes drift through the aisle, with a thin dusting on the floor and the racks still readable."
    ),
    ("warehouse", "snow", "heavy"): (
        "Heavy snow fills the aisle, visibility about 20 m, flakes veil the far racks, and snow dusts the pallets and the floor."
    ),
}


def prompt_for(scene: str, condition: dict) -> str:
    """Photoreal Cosmos Transfer prompt. Same layout; only weather, light, and visibility change."""
    if scene not in _LAYOUT:
        raise ValueError(f"scene must be one of {SCENES}, got {scene!r}")
    try:
        weather = condition["weather"]
        time = condition["time"]
        intensity = condition["intensity"]
    except KeyError as e:
        raise ValueError("condition needs weather, time, and intensity") from e
    if weather not in WEATHERS or time not in TIMES or intensity not in INTENSITIES:
        raise ValueError(f"unknown condition {condition!r}")
    sentences = (
        _LAYOUT[scene],
        _LIGHT[(scene, time, intensity)],
        _WEATHER[(scene, weather, intensity)],
    )
    return " ".join(sentences)


def main():
    print(f"NEGATIVE: {NEGATIVE}")
    print()
    for scene in SCENES:
        for cond in CONDITIONS:
            tag = f"{cond['weather']}_{cond['time']}_{cond['intensity']}"
            print(f"## {scene} / {tag}")
            print(prompt_for(scene, cond))
            print()


if __name__ == "__main__":
    main()
