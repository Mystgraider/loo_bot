from stl_telegram import build_message
from stl_pares import load_history


def test_message_is_slot_specific_and_disclaims_guarantees():
    history = load_history("data/stl_aklan_pares.csv")
    message = build_message(history, "7:00 PM")
    assert "AKLAN STL PARES" in message
    assert "7:00 PM" in message
    assert "guarantee" in message.lower()
    assert "2:00 PM" not in message
