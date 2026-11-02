"""Shared label palette for 2D and 3D bead previews."""

BEAD_COLORS = (
    "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea",
    "#0891b2", "#ea580c", "#4f46e5", "#be123c", "#0f766e",
)


def bead_color(label: str) -> str:
    return BEAD_COLORS[sum(ord(char) for char in label) % len(BEAD_COLORS)]
