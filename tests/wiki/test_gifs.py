import struct
from pathlib import Path

import pytest

from infovore.wiki.gifs import (
    BADGES,
    FONT,
    Canvas,
    background,
    badge,
    counter,
    divider,
    encode,
    new_sticker,
    spinner,
    text_width,
    under_construction,
    write_gifs,
)


def decode(data: bytes) -> tuple[int, int, list[tuple[int, int, int]], list[list[int]]]:
    assert data[:6] == b"GIF89a" and data[-1] == 0x3B
    width, height, packed = struct.unpack("<HHB", data[6:11])
    bits = (packed & 7) + 1
    pos = 13
    palette = [tuple(data[pos + 3 * i : pos + 3 * i + 3]) for i in range(1 << bits)]
    pos += 3 * (1 << bits)
    frames = []
    while data[pos] != 0x3B:
        if data[pos] == 0x21:
            pos += 2
            while data[pos]:
                pos += data[pos] + 1
            pos += 1
            continue
        assert data[pos] == 0x2C
        pos += 10
        min_code = data[pos]
        pos += 1
        stream = bytearray()
        while data[pos]:
            stream += data[pos + 1 : pos + 1 + data[pos]]
            pos += data[pos] + 1
        pos += 1
        clear, end, cw = 1 << min_code, (1 << min_code) + 1, min_code + 1
        acc = nbits = 0
        pixels: list[int] = []
        for byte in stream:
            acc |= byte << nbits
            nbits += 8
            while nbits >= cw:
                code = acc & ((1 << cw) - 1)
                acc >>= cw
                nbits -= cw
                if code == end:
                    break
                if code != clear:
                    assert code < clear, "encoder must never emit table codes"
                    pixels.append(code)
        assert len(pixels) == width * height
        frames.append(pixels)
    return width, height, [tuple(int(v) for v in p) for p in palette], frames  # type: ignore[misc]


def test_encode_round_trips_pixels_palette_and_frames() -> None:
    palette = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
    frame_a = [0, 1, 2, 1, 2, 0]
    frame_b = [2, 2, 2, 1, 1, 1]
    data = encode(3, 2, palette, [frame_a, frame_b], delay=5)
    width, height, table, frames = decode(data)
    assert (width, height) == (3, 2)
    assert table[:3] == palette and len(table) == 4
    assert frames == [frame_a, frame_b]
    assert b"NETSCAPE2.0" in data and data.count(b"\x21\xf9\x04\x00\x05\x00") == 2
    assert b"NETSCAPE2.0" not in encode(3, 2, palette, [frame_a])
    long = encode(300, 1, palette, [[1] * 300])
    assert decode(long)[3] == [[1] * 300]
    assert encode(3, 2, palette, [frame_a]) == encode(3, 2, palette, [frame_a])


def test_canvas_draws_text_from_the_font() -> None:
    c = Canvas(7, 9)
    c.text(1, 1, "a", 1)
    rows = ["".join("#" if c.pixels[y * 7 + x] else "." for x in range(7)) for y in range(9)]
    assert rows == [
        ".......",
        "..###..",
        ".#...#.",
        ".#...#.",
        ".#####.",
        ".#...#.",
        ".#...#.",
        ".#...#.",
        ".......",
    ]
    assert text_width("NEW!") == 23 and text_width("AB", 2) == 22
    assert all(len(rows) == 7 and all(len(r) == 5 for r in rows) for rows in FONT.values())
    c.put(-1, 0, 1)
    c.put(7, 0, 1)
    big = Canvas(4, 4)
    big.text(0, 0, "?", 1)
    assert not any(big.pixels)


@pytest.mark.parametrize(
    ("data", "size", "frame_count"),
    [
        (background(), (48, 48), 2),
        (under_construction(), (160, 40), 4),
        (spinner(), (24, 24), 8),
        (new_sticker(), (32, 16), 2),
        (divider(), (400, 6), 8),
        (badge("NETSCAPE", "NOW!", (255, 255, 255), (0, 0, 128), (192, 192, 192)), (88, 31), 1),
        (counter(75369), (8 * 16 + 2, 22), 1),
    ],
)
def test_builders_make_valid_gifs(data: bytes, size: tuple[int, int], frame_count: int) -> None:
    width, height, _, frames = decode(data)
    assert (width, height) == size and len(frames) == frame_count
    assert len(set(map(tuple, frames))) == frame_count
    assert len(data) < 60_000


def test_write_gifs_writes_every_file(tmp_path: Path) -> None:
    written = write_gifs(tmp_path, 75369)
    assert written == sorted(
        ["bg.gif", "construction.gif", "spinner.gif", "new.gif", "divider.gif", "counter.gif"]
        + [b[0] for b in BADGES]
    )
    assert all((tmp_path / n).read_bytes().startswith(b"GIF89a") for n in written)
    assert decode((tmp_path / "counter.gif").read_bytes())[3] != decode(counter(1))[3]
