"""
Config ng mga PCSO games na susuriin.
type:
  "combo"  -> pumipili ng `pick` UNIQUE na numero mula sa range (walang ulit), gaya ng 6/55
  "digit"  -> pumipili ng `pick` digit, PUWEDENG magulit (gaya ng Swertres na 0-9 bawat posisyon)

ordered:
  True  -> may pakialam sa PAGKAKASUNOD-SUNOD ang official standard bet (EZ2, Swertres,
           4D, 6D -- lahat ng "digit" games ay ganito talaga sa PCSO rules). Kung gusto
           ng manlalaro na hindi mahalaga ang order, gagamitin ang "Rambolito" system
           (mas mababang panalo, pero mas madaling tumama).
  False -> hindi mahalaga ang pagkakasunod-sunod (lahat ng 6-number lotto games)

draw_days:
  Listahan ng weekday kung kailan aktwal na nagdo-draw ang laro (galing sa historical
  data). None = araw-araw (gaya ng EZ2/Swertres na 3x/araw).

I-verify/i-adjust ang mga value dito kung mag-iiba ang official rules ng PCSO.
"""

GAMES = {
    "6/55": {"type": "combo", "pick": 6, "range": (1, 55), "csv": "data/draws_6_55.csv",
             "ordered": False, "draw_days": ["Monday", "Wednesday", "Saturday"]},
    "6/58": {"type": "combo", "pick": 6, "range": (1, 58), "csv": "data/draws_6_58.csv",
             "ordered": False, "draw_days": ["Sunday", "Tuesday", "Friday"]},
    "6/49": {"type": "combo", "pick": 6, "range": (1, 49), "csv": "data/draws_6_49.csv",
             "ordered": False, "draw_days": ["Sunday", "Tuesday", "Thursday"]},
    "6/45": {"type": "combo", "pick": 6, "range": (1, 45), "csv": "data/draws_6_45.csv",
             "ordered": False, "draw_days": ["Monday", "Wednesday", "Friday"]},
    "6/42": {"type": "combo", "pick": 6, "range": (1, 42), "csv": "data/draws_6_42.csv",
             "ordered": False, "draw_days": ["Tuesday", "Thursday", "Saturday"]},
    "ez2":  {"type": "combo", "pick": 2, "range": (1, 31), "csv": "data/draws_ez2.csv",
             "ordered": True, "draw_days": None, "multi_draw_per_day": True},
    "swertres": {"type": "digit", "pick": 3, "range": (0, 9), "csv": "data/draws_swertres.csv",
                 "ordered": True, "draw_days": None, "multi_draw_per_day": True},
    "6d": {"type": "digit", "pick": 6, "range": (0, 9), "csv": "data/draws_6d.csv",
           "ordered": True, "draw_days": ["Tuesday", "Thursday", "Saturday"]},
    "4d": {"type": "digit", "pick": 4, "range": (0, 9), "csv": "data/draws_4d.csv",
           "ordered": True, "draw_days": ["Monday", "Wednesday", "Friday"]},
}

# Minimum na bilang ng historical draws bago paganahin ang ML component.
# Sa mababa sa 'to, mas malaking chance na mag-overfit lang ang model sa noise.
MIN_DRAWS_FOR_ML = 60

# Bilang ng huling draws na susuriin (per instructions mo: 500 max)
MAX_DRAWS_WINDOW = 500

# Order ng draws kada araw para sa "multi_draw_per_day" games (EZ2, Swertres).
# TANDAAN: umaasa ito na palaging 2PM->5PM->9PM ang pagkakasunod-sunod ng
# 3 rows kada petsa sa CSV (ito ang order na ginagamit ng scraper.py at ng
# orihinal na historical data import).
SLOT_LABELS = ["2:00 PM", "5:00 PM", "9:00 PM"]
