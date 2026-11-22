"""A small QR code encoder for the two-step sign-in setup (walkthrough, 2026-11-16).

The setup key was shown only as 32 characters to type into an authenticator
app, and the owner's codes were refused twice: a key typed by hand is easy to
get wrong by one character, and the server cannot tell a mistyped key from a
wrong code. Scanning a QR code removes the typing.

The code is drawn here, on the server, as inline SVG. The secret never leaves
WarriorIQ: no QR web service or image CDN is involved, and no new package is
installed for it. The encoder covers what an otpauth:// address needs: byte
mode, error correction level M, versions 1 to 20 (up to 666 bytes). It follows
ISO/IEC 18004 and is checked in tests/test_two_factor.py by decoding the result
with OpenCV's QR reader.
"""

from __future__ import annotations

# Error correction level M: (EC codewords per block, [(blocks, data codewords per block), ...])
_BLOCKS_M = {
    1: (10, [(1, 16)]), 2: (16, [(1, 28)]), 3: (26, [(1, 44)]), 4: (18, [(2, 32)]),
    5: (24, [(2, 43)]), 6: (16, [(4, 27)]), 7: (18, [(4, 31)]), 8: (22, [(2, 38), (2, 39)]),
    9: (22, [(3, 36), (2, 37)]), 10: (26, [(4, 43), (1, 44)]), 11: (30, [(1, 50), (4, 51)]),
    12: (22, [(6, 36), (2, 37)]), 13: (22, [(8, 37), (1, 38)]), 14: (24, [(4, 40), (5, 41)]),
    15: (24, [(5, 41), (5, 42)]), 16: (28, [(7, 45), (3, 46)]), 17: (28, [(10, 46), (1, 47)]),
    18: (26, [(9, 43), (4, 44)]), 19: (26, [(3, 44), (11, 45)]), 20: (26, [(3, 41), (13, 42)]),
}
_ALIGNMENT = {
    1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34], 7: [6, 22, 38],
    8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50], 11: [6, 30, 54], 12: [6, 32, 58],
    13: [6, 34, 62], 14: [6, 26, 46, 66], 15: [6, 26, 48, 70], 16: [6, 26, 50, 74],
    17: [6, 30, 54, 78], 18: [6, 30, 56, 82], 19: [6, 30, 58, 86], 20: [6, 34, 62, 90],
}
_FORMAT_BITS_M = 0  # the two error-correction bits for level M
MAX_VERSION = max(_BLOCKS_M)


def _data_capacity(version: int) -> int:
    return sum(count * size for count, size in _BLOCKS_M[version][1])


def _gf_mul(x: int, y: int) -> int:
    z = 0
    for i in reversed(range(8)):
        z = (z << 1) ^ ((z >> 7) * 0x11D)
        z ^= ((y >> i) & 1) * x
    return z


def _rs_divisor(degree: int) -> list[int]:
    result = [0] * (degree - 1) + [1]
    root = 1
    for _ in range(degree):
        for j in range(degree):
            result[j] = _gf_mul(result[j], root)
            if j + 1 < degree:
                result[j] ^= result[j + 1]
        root = _gf_mul(root, 0x02)
    return result


def _rs_remainder(data: list[int], divisor: list[int]) -> list[int]:
    result = [0] * len(divisor)
    for byte in data:
        factor = byte ^ result.pop(0)
        result.append(0)
        for i, coefficient in enumerate(divisor):
            result[i] ^= _gf_mul(coefficient, factor)
    return result


def _codewords(payload: bytes) -> tuple[int, list[int]]:
    """(version, final interleaved codewords) for the smallest version that fits."""
    for version in range(1, MAX_VERSION + 1):
        count_bits = 8 if version < 10 else 16
        capacity_bits = _data_capacity(version) * 8
        if 4 + count_bits + len(payload) * 8 <= capacity_bits:
            break
    else:
        raise ValueError("too much data for a QR code of version 20 or below")
    bits: list[int] = []

    def put(value: int, length: int) -> None:
        bits.extend((value >> i) & 1 for i in reversed(range(length)))

    put(0b0100, 4)                        # byte mode
    put(len(payload), count_bits)
    for byte in payload:
        put(byte, 8)
    put(0, min(4, capacity_bits - len(bits)))            # terminator
    put(0, -len(bits) % 8)                               # to a whole byte
    data = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]
    pad = 0xEC
    while len(data) < capacity_bits // 8:
        data.append(pad)
        pad ^= 0xEC ^ 0x11

    ec_len, groups = _BLOCKS_M[version]
    divisor = _rs_divisor(ec_len)
    blocks, offset = [], 0
    for count, size in groups:
        for _ in range(count):
            chunk = data[offset:offset + size]
            offset += size
            blocks.append((chunk, _rs_remainder(chunk, divisor)))
    result: list[int] = []
    for i in range(max(len(chunk) for chunk, _ in blocks)):
        result.extend(chunk[i] for chunk, _ in blocks if i < len(chunk))
    for i in range(ec_len):
        result.extend(ec[i] for _, ec in blocks)
    return version, result


class _Matrix:
    def __init__(self, version: int):
        self.version = version
        self.size = version * 4 + 17
        self.modules = [[False] * self.size for _ in range(self.size)]
        self.function = [[False] * self.size for _ in range(self.size)]

    def set_function(self, x: int, y: int, dark: bool) -> None:
        self.modules[y][x] = dark
        self.function[y][x] = True

    def draw_function_patterns(self) -> None:
        size = self.size
        for i in range(size):                                      # timing
            self.set_function(6, i, i % 2 == 0)
            self.set_function(i, 6, i % 2 == 0)
        for cx, cy in ((3, 3), (size - 4, 3), (3, size - 4)):     # finders and separators
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < size and 0 <= y < size:
                        self.set_function(x, y, max(abs(dx), abs(dy)) not in (2, 4))
        positions = _ALIGNMENT[self.version]
        last = len(positions) - 1
        for i, ax in enumerate(positions):
            for j, ay in enumerate(positions):
                if (i, j) in ((0, 0), (0, last), (last, 0)):
                    continue
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        self.set_function(ax + dx, ay + dy, max(abs(dx), abs(dy)) != 1)
        self.draw_format(0)                                        # reserve; redrawn later
        if self.version >= 7:
            remainder = self.version
            for _ in range(12):
                remainder = (remainder << 1) ^ ((remainder >> 11) * 0x1F25)
            bits = self.version << 12 | remainder
            for i in range(18):
                dark = bool((bits >> i) & 1)
                a, b = size - 11 + i % 3, i // 3
                self.set_function(a, b, dark)
                self.set_function(b, a, dark)

    def draw_format(self, mask: int) -> None:
        data = _FORMAT_BITS_M << 3 | mask
        remainder = data
        for _ in range(10):
            remainder = (remainder << 1) ^ ((remainder >> 9) * 0x537)
        bits = (data << 10 | remainder) ^ 0x5412

        def bit(i: int) -> bool:
            return bool((bits >> i) & 1)

        size = self.size
        for i in range(6):
            self.set_function(8, i, bit(i))
        self.set_function(8, 7, bit(6))
        self.set_function(8, 8, bit(7))
        self.set_function(7, 8, bit(8))
        for i in range(9, 15):
            self.set_function(14 - i, 8, bit(i))
        for i in range(8):
            self.set_function(size - 1 - i, 8, bit(i))
        for i in range(8, 15):
            self.set_function(8, size - 15 + i, bit(i))
        self.set_function(8, size - 8, True)                       # the dark module

    def draw_data(self, codewords: list[int]) -> None:
        size, i, total = self.size, 0, len(codewords) * 8
        right = size - 1
        while right >= 1:
            if right == 6:
                right = 5
            upward = ((right + 1) & 2) == 0
            for vertical in range(size):
                y = size - 1 - vertical if upward else vertical
                for j in range(2):
                    x = right - j
                    if not self.function[y][x] and i < total:
                        self.modules[y][x] = bool((codewords[i >> 3] >> (7 - (i & 7))) & 1)
                        i += 1
            right -= 2

    def apply_mask(self, mask: int) -> None:
        rule = _MASKS[mask]
        for y in range(self.size):
            for x in range(self.size):
                if not self.function[y][x] and rule(x, y):
                    self.modules[y][x] = not self.modules[y][x]

    def penalty(self) -> int:
        size, grid, score = self.size, self.modules, 0
        lines = grid + [[grid[y][x] for y in range(size)] for x in range(size)]
        for line in lines:                                         # runs of five or more
            run_colour, run = line[0], 1
            for value in line[1:]:
                if value == run_colour:
                    run += 1
                else:
                    if run >= 5:
                        score += run - 2
                    run_colour, run = value, 1
            if run >= 5:
                score += run - 2
            text = "".join("1" if v else "0" for v in line)          # finder look-alikes
            score += 40 * (text.count("10111010000") + text.count("00001011101"))
        for y in range(size - 1):                                  # 2x2 blocks
            for x in range(size - 1):
                if grid[y][x] == grid[y][x + 1] == grid[y + 1][x] == grid[y + 1][x + 1]:
                    score += 3
        dark = sum(sum(row) for row in grid)                       # balance
        score += 10 * (abs(dark * 20 - size * size * 10) // (size * size))
        return score


_MASKS = (
    lambda x, y: (x + y) % 2 == 0,
    lambda x, y: y % 2 == 0,
    lambda x, y: x % 3 == 0,
    lambda x, y: (x + y) % 3 == 0,
    lambda x, y: (x // 3 + y // 2) % 2 == 0,
    lambda x, y: x * y % 2 + x * y % 3 == 0,
    lambda x, y: (x * y % 2 + x * y % 3) % 2 == 0,
    lambda x, y: ((x + y) % 2 + x * y % 3) % 2 == 0,
)


def matrix(text: str) -> list[list[bool]]:
    """The QR code for text as rows of modules (True is dark), without the quiet zone."""
    version, codewords = _codewords(text.encode("utf-8"))
    best, best_score = None, None
    for mask in range(8):
        grid = _Matrix(version)
        grid.draw_function_patterns()
        grid.draw_data(codewords)
        grid.apply_mask(mask)
        grid.draw_format(mask)
        score = grid.penalty()
        if best_score is None or score < best_score:
            best, best_score = grid, score
    return best.modules


def svg(text: str, *, module_px: int = 5, quiet: int = 4, label: str = "QR code") -> str:
    """An SVG image of the QR code: dark modules on white, with the standard quiet zone.

    Always black on white whatever the page's theme: phone cameras read a light
    code on a dark page unreliably.
    """
    from html import escape

    modules = matrix(text)
    size = len(modules) + quiet * 2
    path = "".join(
        f"M{x + quiet},{y + quiet}h1v1h-1z"
        for y, row in enumerate(modules) for x, dark in enumerate(row) if dark
    )
    pixels = size * module_px
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{pixels}" '
            f'height="{pixels}" role="img" aria-label="{escape(label)}" shape-rendering="crispEdges">'
            f'<rect width="{size}" height="{size}" fill="#fff"/><path d="{path}" fill="#000"/></svg>')
