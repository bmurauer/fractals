from __future__ import annotations  # forward type hints

from textwrap import wrap


class Color:
    def __init__(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b

    @classmethod
    def from_hex(cls, hex_string: str) -> Color:
        red, green, blue = wrap(hex_string, 2)
        return Color(int(red, 16), int(green, 16), int(blue, 16))

    def interpolate_towards(self, other: Color, fraction: float) -> Color:
        assert 0.0 <= fraction <= 1.0
        r = round(self.r + (other.r - self.r) * fraction)
        g = round(self.g + (other.g - self.g) * fraction)
        b = round(self.b + (other.b - self.b) * fraction)
        return Color(r, g, b)

    def __repr__(self):
        return f"{self.r:02x}{self.g:02x}{self.b:02x}"
