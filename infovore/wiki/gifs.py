import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

Color = tuple[int, int, int]
Frame = Sequence[int]

BLACK: Final = (0, 0, 0)
WHITE: Final = (255, 255, 255)
YELLOW: Final = (255, 255, 0)
RED: Final = (255, 0, 0)
LIME: Final = (0, 255, 0)
BLUE: Final = (0, 0, 255)
NAVY: Final = (0, 0, 128)
CYAN: Final = (0, 255, 255)
MAGENTA: Final = (255, 0, 255)
GREY: Final = (192, 192, 192)
DARK: Final = (64, 64, 64)
ORANGE: Final = (255, 128, 0)
RAINBOW: Final = (RED, ORANGE, YELLOW, LIME, CYAN, BLUE, MAGENTA, (128, 0, 255))

FONT: Final[Mapping[str, Sequence[str]]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "10001", "11001", "10101", "10011", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
}
GLYPH_W: Final = 5
GLYPH_H: Final = 7
ADVANCE: Final = 6


class Canvas:
    def __init__(self, width: int, height: int, fill: int = 0) -> None:
        self.width, self.height = width, height
        self.pixels = [fill] * (width * height)

    def put(self, x: int, y: int, color: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y * self.width + x] = color

    def rect(self, x: int, y: int, w: int, h: int, color: int) -> None:
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self.put(xx, yy, color)

    def border(self, color: int, thickness: int = 1) -> None:
        for t in range(thickness):
            self.rect(t, t, self.width - 2 * t, 1, color)
            self.rect(t, self.height - 1 - t, self.width - 2 * t, 1, color)
            self.rect(t, t, 1, self.height - 2 * t, color)
            self.rect(self.width - 1 - t, t, 1, self.height - 2 * t, color)

    def text(self, x: int, y: int, string: str, color: int, scale: int = 1) -> None:
        for i, ch in enumerate(string.upper()):
            rows = FONT.get(ch, FONT[" "])
            for r, row in enumerate(rows):
                for c, bit in enumerate(row):
                    if bit == "1":
                        self.rect(x + (i * ADVANCE + c) * scale, y + r * scale, scale, scale, color)

    def centered(self, y: int, string: str, color: int, scale: int = 1) -> None:
        width = (len(string) * ADVANCE - 1) * scale
        self.text((self.width - width) // 2, y, string, color, scale)


def text_width(string: str, scale: int = 1) -> int:
    return (len(string) * ADVANCE - 1) * scale


def _lzw(indices: Iterable[int], min_code: int) -> bytes:
    clear, end = 1 << min_code, (1 << min_code) + 1
    width, span = min_code + 1, (1 << min_code) - 2
    acc = nbits = count = 0
    out = bytearray()

    def emit(code: int) -> None:
        nonlocal acc, nbits
        acc |= code << nbits
        nbits += width
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8

    emit(clear)
    for index in indices:
        if count == span:
            emit(clear)
            count = 0
        emit(index)
        count += 1
    emit(end)
    if nbits:
        out.append(acc & 0xFF)
    blocks = bytearray()
    for i in range(0, len(out), 255):
        chunk = out[i : i + 255]
        blocks += bytes([len(chunk)]) + chunk
    return bytes(blocks) + b"\x00"


def encode(
    width: int, height: int, palette: Sequence[Color], frames: Sequence[Frame], delay: int = 10
) -> bytes:
    bits = max(2, (len(palette) - 1).bit_length())
    table = list(palette) + [BLACK] * ((1 << bits) - len(palette))
    out = bytearray(b"GIF89a")
    out += struct.pack("<HHBBB", width, height, 0x80 | ((bits - 1) << 4) | (bits - 1), 0, 0)
    out += b"".join(bytes(c) for c in table)
    if len(frames) > 1:
        out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"
    for frame in frames:
        out += b"\x21\xf9\x04\x00" + struct.pack("<HB", delay, 0) + b"\x00"
        out += b"\x2c" + struct.pack("<HHHHB", 0, 0, width, height, 0)
        out += bytes([bits]) + _lzw(frame, bits)
    return bytes(out + b"\x3b")


def background() -> bytes:
    palette = (BLACK, WHITE, GREY, YELLOW, CYAN, DARK)
    stars = [
        (3, 5),
        (17, 2),
        (29, 11),
        (9, 20),
        (40, 19),
        (22, 31),
        (36, 38),
        (6, 42),
        (45, 7),
        (61, 14),
        (74, 3),
        (88, 27),
        (53, 36),
        (69, 48),
        (83, 57),
        (58, 66),
        (91, 80),
        (12, 61),
        (27, 74),
        (44, 88),
        (66, 83),
        (2, 90),
        (79, 93),
        (33, 55),
        (50, 9),
        (95, 44),
        (20, 95),
    ]
    frames = []
    for phase in range(2):
        c = Canvas(96, 96)
        for i, (x, y) in enumerate(stars):
            c.put(x, y, 1 + (i * 7 + phase) % 5)
        frames.append(c.pixels)
    return encode(96, 96, palette, frames, delay=90)


def under_construction() -> bytes:
    palette = (BLACK, YELLOW, RED)
    frames = []
    for shift in range(0, 8, 2):
        c = Canvas(160, 40, 1)
        for y in list(range(0, 8)) + list(range(32, 40)):
            for x in range(160):
                if (x + y + shift) % 8 < 4:
                    c.put(x, y, 0)
        c.centered(13, "UNDER CONSTRUCTION", 0)
        c.rect(0, 24, 160, 1, 2)
        frames.append(c.pixels)
    return encode(160, 40, palette, frames, delay=12)


def spinner() -> bytes:
    palette = (BLACK, LIME, (0, 160, 0), (0, 80, 0), GREY)
    dots = [(12, 3), (18, 5), (21, 11), (18, 17), (12, 19), (6, 17), (3, 11), (6, 5)]
    frames = []
    for phase in range(8):
        c = Canvas(24, 24)
        c.border(4)
        for i, (x, y) in enumerate(dots):
            age = (phase - i) % 8
            c.rect(x - 1, y - 1, 3, 3, 1 if age == 0 else 2 if age == 1 else 3)
        frames.append(c.pixels)
    return encode(24, 24, palette, frames, delay=8)


def new_sticker() -> bytes:
    palette = (RED, YELLOW)
    frames = []
    for fg, bg in ((0, 1), (1, 0)):
        c = Canvas(32, 16, bg)
        c.border(fg)
        c.centered(4, "NEW!", fg)
        frames.append(c.pixels)
    return encode(32, 16, palette, frames, delay=40)


def divider() -> bytes:
    frames = []
    for shift in range(8):
        c = Canvas(400, 6)
        for x in range(400):
            c.rect(x, 0, 1, 6, (x // 8 + shift) % 8)
        frames.append(c.pixels)
    return encode(400, 6, RAINBOW, frames, delay=10)


def badge(top: str, bottom: str, fg: Color, bg: Color, trim: Color) -> bytes:
    c = Canvas(88, 31, 1)
    c.border(2, 2)
    c.centered(7, top, 0)
    c.centered(17, bottom, 0)
    return encode(88, 31, (fg, bg, trim), [c.pixels])


def counter(number: int, digits: int = 8) -> bytes:
    text = f"{number:0{digits}d}"[-digits:]
    cell = GLYPH_W * 2 + 6
    c = Canvas(cell * digits + 2, GLYPH_H * 2 + 8, 1)
    for i, ch in enumerate(text):
        x = 1 + i * cell
        c.rect(x, 1, cell, GLYPH_H * 2 + 6, 0)
        c.rect(x, 1, 1, GLYPH_H * 2 + 6, 1)
        c.text(x + 3, 4, ch, 2, 2)
    return encode(c.width, c.height, (BLACK, GREY, WHITE), [c.pixels])


BADGES: Final = (
    ("badge-netscape.gif", "NETSCAPE", "NOW!", WHITE, NAVY, GREY),
    ("badge-yahoo.gif", "YAHOO!", "LISTED", WHITE, (128, 0, 128), YELLOW),
    ("badge-vi.gif", "MADE", "WITH VI", LIME, BLACK, LIME),
    ("badge-irix.gif", "GET", "IRIX 6.5", WHITE, (0, 64, 128), CYAN),
    ("badge-html.gif", "VALID", "HTML 3.2", BLACK, YELLOW, RED),
    ("badge-800x600.gif", "800X600", "OR BUST", YELLOW, RED, WHITE),
)


def write_gifs(out: Path, claims: int) -> list[str]:
    files: dict[str, bytes] = {
        "bg.gif": background(),
        "construction.gif": under_construction(),
        "spinner.gif": spinner(),
        "new.gif": new_sticker(),
        "divider.gif": divider(),
        "counter.gif": counter(claims),
    }
    for name, top, bottom, fg, bg, trim in BADGES:
        files[name] = badge(top, bottom, fg, bg, trim)
    for name, data in files.items():
        (out / name).write_bytes(data)
    return sorted(files)
