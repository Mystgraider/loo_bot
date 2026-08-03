"""
Config ng mga PCSO games na susuriin.
type:
  "combo"  -> pumipili ng `pick` UNIQUE na numero mula sa range (walang ulit), gaya ng 6/55
  "digit"  -> pumipili ng `pick` digit, PUWEDENG magulit (gaya ng Swertres na 0-9 bawat posisyon)

I-verify/i-adjust ang mga range dito kung mag-iiba ang official rules ng PCSO.
"""

GAMES = {
    "6/55": {"type": "combo", "pick": 6, "range": (1, 55), "csv": "data/draws_6_55.csv"},
    "6/58": {"type": "combo", "pick": 6, "range": (1, 58), "csv": "data/draws_6_58.csv"},
    "6/49": {"type": "combo", "pick": 6, "range": (1, 49), "csv": "data/draws_6_49.csv"},
    "6/45": {"type": "combo", "pick": 6, "range": (1, 45), "csv": "data/draws_6_45.csv"},
    "6/42": {"type": "combo", "pick": 6, "range": (1, 42), "csv": "data/draws_6_42.csv"},
    "ez2":  {"type": "combo", "pick": 2, "range": (1, 31), "csv": "data/draws_ez2.csv"},
    "swertres": {"type": "digit", "pick": 3, "range": (0, 9), "csv": "data/draws_swertres.csv"},
    "6d": {"type": "digit", "pick": 6, "range": (0, 9), "csv": "data/draws_6d.csv"},
    "4d": {"type": "digit", "pick": 4, "range": (0, 9), "csv": "data/draws_4d.csv"},
}

# Minimum na bilang ng historical draws bago paganahin ang ML component.
# Sa mababa sa 'to, mas malaking chance na mag-overfit lang ang model sa noise.
MIN_DRAWS_FOR_ML = 60

# Bilang ng huling draws na susuriin (per instructions mo: 500 max)
MAX_DRAWS_WINDOW = 500
