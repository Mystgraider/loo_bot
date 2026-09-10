"""Central configuration for all supported PCSO games."""

GAMES = {
    "6/55": {
        "type": "combo", "pick": 6, "range": (1, 55),
        "csv": "data/draws_6_55.csv", "order_matters": False,
        "allow_repetition": False,
        "draw_days": ["Monday", "Wednesday", "Saturday"],
    },
    "6/58": {
        "type": "combo", "pick": 6, "range": (1, 58),
        "csv": "data/draws_6_58.csv", "order_matters": False,
        "allow_repetition": False,
        "draw_days": ["Sunday", "Tuesday", "Friday"],
    },
    "6/49": {
        "type": "combo", "pick": 6, "range": (1, 49),
        "csv": "data/draws_6_49.csv", "order_matters": False,
        "allow_repetition": False,
        "draw_days": ["Sunday", "Tuesday", "Thursday"],
    },
    "6/45": {
        "type": "combo", "pick": 6, "range": (1, 45),
        "csv": "data/draws_6_45.csv", "order_matters": False,
        "allow_repetition": False,
        "draw_days": ["Monday", "Wednesday", "Friday"],
    },
    "6/42": {
        "type": "combo", "pick": 6, "range": (1, 42),
        "csv": "data/draws_6_42.csv", "order_matters": False,
        "allow_repetition": False,
        "draw_days": ["Tuesday", "Thursday", "Saturday"],
    },
    "ez2": {
        "type": "combo", "pick": 2, "range": (1, 31),
        "csv": "data/draws_ez2.csv", "order_matters": True,
        "allow_repetition": False, "multi_draw_per_day": True,
    },
    "swertres": {
        "type": "digit", "pick": 3, "range": (0, 9),
        "csv": "data/draws_swertres.csv", "order_matters": True,
        "allow_repetition": True, "multi_draw_per_day": True,
    },
    "6d": {
        "type": "digit", "pick": 6, "range": (0, 9),
        "csv": "data/draws_6d.csv", "order_matters": True,
        "allow_repetition": True,
        "draw_days": ["Tuesday", "Thursday", "Saturday"],
    },
    "4d": {
        "type": "digit", "pick": 4, "range": (0, 9),
        "csv": "data/draws_4d.csv", "order_matters": True,
        "allow_repetition": True,
        "draw_days": ["Monday", "Wednesday", "Friday"],
    },
}

MIN_DRAWS_FOR_ML = 60
MAX_DRAWS_WINDOW = 500
SLOT_LABELS = ["2:00 PM", "5:00 PM", "9:00 PM"]
