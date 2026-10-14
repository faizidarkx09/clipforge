"""Karaoke ASS subtitle generation.

Approach: true ASS karaoke with {\\k<centiseconds>} tags. Each caption line
is a single Dialogue event whose words carry \\k durations; the renderer
sweeps each word from the style's SecondaryColour (unsung) to PrimaryColour
(sung) as it is spoken. For the "hormozi" preset that means: white words
turning bold yellow exactly on the beat — no per-word event bookkeeping.

8 presets:
    hormozi     - bold yellow highlight on white, lower third, 3-5 words/line
    minimal     - small white text, subtle
    neon        - cyan glow highlight
    karaoke-box - opaque dark box behind the karaoke words
    pop         - huge ALL-CAPS hot-pink words, thick outline
    typewriter  - Courier New, steady white text (karaoke sweep invisible)
    elegant     - Georgia italic with a gold highlight
    outline     - huge white words with a thick black outline
PlayRes is 1080x1920 (vertical shorts).

Style dicts carry the ASS fields plus two optional flags honored by
write_ass(): "uppercase" (render words in ALL CAPS) and "italic"
(lowercase alias for Italic=-1).
"""
from __future__ import annotations

import os

# ASS colors are &HAABBGGRR.
WHITE = "&H00FFFFFF"
YELLOW = "&H0000FFFF"   # BGR: 00 FF FF
CYAN = "&H00FFFF00"      # BGR: FF FF 00
BLACK = "&H00000000"
GREY = "&H00AAAAAA"
SHADOW = "&H99000000"
DARK_BOX = "&HCC000000"  # semi-opaque black, used for BorderStyle 3 boxes

# Keyword -> emoji appended after the word when the "emoji" caption override
# is enabled (matched case-insensitively, punctuation stripped).
EMOJI_MAP = {
    "money": "💰", "cash": "💰", "dollar": "💵", "dollars": "💵",
    "rich": "💰", "profit": "💰",
    "fire": "🔥", "lit": "🔥",
    "love": "❤️", "heart": "❤️",
    "win": "🏆", "winner": "🏆", "success": "🏆",
    "secret": "🤫", "secrets": "🤫",
    "crazy": "🤯", "insane": "🤯",
    "free": "🎁", "gift": "🎁",
    "new": "✨",
    "fast": "⚡", "quick": "⚡", "speed": "⚡",
    "warning": "⚠️", "stop": "🛑",
    "star": "⭐", "idea": "💡", "phone": "📱", "car": "🚗",
    "happy": "😄", "sad": "😢",
}


def _hex_to_ass(hex_color: str) -> str:
    """'#RRGGBB' -> ASS '&H00BBGGRR' (full opacity)."""
    h = hex_color.strip().lstrip("#")
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"Bad hex color {hex_color!r}; expected '#RRGGBB'")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()

STYLES = {
    "hormozi": {
        "Fontname": "Arial",
        "Fontsize": 76,
        "Bold": -1,
        "PrimaryColour": YELLOW,      # sung word -> bold yellow
        "SecondaryColour": WHITE,     # upcoming words -> white
        "OutlineColour": BLACK,
        "BackColour": SHADOW,
        "BorderStyle": 1,
        "Outline": 2,
        "Shadow": 1,
        "Alignment": 2,               # bottom-center (lower third)
        "MarginV": 150,
    },
    "minimal": {
        "Fontname": "Arial",
        "Fontsize": 52,
        "Bold": 0,
        "PrimaryColour": WHITE,
        "SecondaryColour": GREY,
        "OutlineColour": BLACK,
        "BackColour": SHADOW,
        "BorderStyle": 1,
        "Outline": 1,
        "Shadow": 1,
        "Alignment": 2,
        "MarginV": 120,
    },
    "neon": {
        "Fontname": "Arial",
        "Fontsize": 70,
        "Bold": -1,
        "PrimaryColour": CYAN,        # sung word -> cyan glow
        "SecondaryColour": WHITE,
        "OutlineColour": BLACK,
        "BackColour": SHADOW,
        "BorderStyle": 1,
        "Outline": 1,
        "Shadow": 0,
        "Alignment": 2,
        "MarginV": 150,
    },
    "karaoke-box": {
        "Fontname": "Arial",
        "Fontsize": 68,
        "Bold": -1,
        "PrimaryColour": YELLOW,      # sung word -> yellow
        "SecondaryColour": WHITE,     # upcoming words -> white
        "OutlineColour": BLACK,
        "BackColour": DARK_BOX,       # opaque box behind the line
        "BorderStyle": 3,
        "Outline": 1,
        "Shadow": 0,
        "Alignment": 2,
        "MarginV": 150,
    },
    "pop": {
        "Fontname": "Arial",
        "Fontsize": 100,
        "Bold": -1,
        "PrimaryColour": _hex_to_ass("#FF2E88"),  # hot pink highlight
        "SecondaryColour": WHITE,
        "OutlineColour": BLACK,
        "BackColour": SHADOW,
        "BorderStyle": 1,
        "Outline": 4,                 # thick outline
        "Shadow": 2,
        "Alignment": 2,
        "MarginV": 150,
        "uppercase": True,            # ALL-CAPS words
    },
    "typewriter": {
        "Fontname": "Courier New",
        # Primary == Secondary: the \k karaoke sweep becomes invisible, so
        # words simply appear in steady white as they are spoken.
        "PrimaryColour": WHITE,
        "SecondaryColour": WHITE,
        "Fontsize": 60,
        "Bold": 0,
        "OutlineColour": BLACK,
        "BackColour": SHADOW,
        "BorderStyle": 1,
        "Outline": 1,
        "Shadow": 1,
        "Alignment": 2,
        "MarginV": 150,
    },
    "elegant": {
        "Fontname": "Georgia",
        "Fontsize": 64,
        "Bold": 0,
        "Italic": -1,
        "PrimaryColour": _hex_to_ass("#FFD700"),  # gold highlight
        "SecondaryColour": WHITE,
        "OutlineColour": BLACK,
        "BackColour": SHADOW,
        "BorderStyle": 1,
        "Outline": 1,
        "Shadow": 1,
        "Alignment": 2,
        "MarginV": 150,
    },
    "outline": {
        "Fontname": "Arial",
        "Fontsize": 110,
        "Bold": -1,
        "PrimaryColour": WHITE,
        "SecondaryColour": GREY,
        "OutlineColour": BLACK,
        "BackColour": SHADOW,
        "BorderStyle": 1,
        "Outline": 4,                 # thick black outline
        "Shadow": 0,
        "Alignment": 2,
        "MarginV": 150,
    },
}

_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {name},{Fontname},{Fontsize},{PrimaryColour},{SecondaryColour},{OutlineColour},{BackColour},{Bold},{Italic},0,0,100,100,0,0,{BorderStyle},{Outline},{Shadow},{Alignment},20,20,{MarginV},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

_DIALOGUE = "Dialogue: 0,{start},{end},{style},,0,0,0,,{text}\n"


def _ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs == 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean(word: str) -> str:
    return word.replace("{", "").replace("}", "").replace("\n", " ").strip()


def _chunk_words(words: list[dict], max_per_line: int = 4) -> list[list[dict]]:
    """Group words into caption lines of 3-5 words.

    Prefer breaking at sentence punctuation; never leave a 1-word orphan.
    """
    lines: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        cur.append(w)
        stripped = w["word"].strip()
        at_punct = stripped and stripped[-1] in ".!?,;:"
        if len(cur) >= max_per_line or (at_punct and len(cur) >= 2):
            lines.append(cur)
            cur = []
    if cur:
        if lines and len(cur) < 2:
            lines[-1].extend(cur)  # merge orphan into previous line
        else:
            lines.append(cur)
    return lines


_EMOJI_STRIP = ".,!?;:'\"()[]-–—"


def write_ass(
    words: list[dict],
    path: str,
    style: str = "hormozi",
    max_words_per_line: int = 4,
    overrides: dict | None = None,
) -> str:
    """Write a karaoke ASS file for `words` ([{start,end,word}, ...]).

    Each word gets a {\\k<cs>} tag with its spoken duration in centiseconds;
    libass/mpv/ffmpeg's subtitles filter render the sweep natively.

    `overrides` (all optional) tweaks the chosen style per render:
        font_size        int -> Fontsize
        position         'top'|'center'|'bottom'
                         -> Alignment 8|5|2, MarginV 120|0|150
        text_color       '#RRGGBB' -> SecondaryColour (unsung words)
        highlight_color  '#RRGGBB' -> PrimaryColour (sung words)
        bg_box           bool -> BorderStyle 3 + dark box (True) or
                         BorderStyle 1 (False)
        emoji            bool -> append EMOJI_MAP emoji after matching words

    Style-dict flags "uppercase" / "italic" (e.g. on the 'pop' style) are
    honored as well. Returns `path`.
    """
    if style not in STYLES:
        raise ValueError(
            f"Unknown caption style {style!r}; choose from {sorted(STYLES)}"
        )
    words = [w for w in words if w.get("word", "").strip()]
    words.sort(key=lambda w: w["start"])

    st = dict(STYLES[style])
    st["name"] = style

    ov = overrides or {}
    if "font_size" in ov and ov["font_size"]:
        st["Fontsize"] = int(ov["font_size"])
    pos = ov.get("position")
    if pos == "top":
        st["Alignment"], st["MarginV"] = 8, 120
    elif pos == "center":
        st["Alignment"], st["MarginV"] = 5, 0
    elif pos == "bottom":
        st["Alignment"], st["MarginV"] = 2, 150
    if ov.get("text_color"):
        st["SecondaryColour"] = _hex_to_ass(ov["text_color"])
    if ov.get("highlight_color"):
        st["PrimaryColour"] = _hex_to_ass(ov["highlight_color"])
    if "bg_box" in ov:
        if ov["bg_box"]:
            st["BorderStyle"], st["BackColour"] = 3, DARK_BOX
        else:
            st["BorderStyle"] = 1

    st.setdefault("Italic", 0)
    if st.get("italic") is True:      # lowercase flag alias for Italic=-1
        st["Italic"] = -1
    uppercase = bool(st.get("uppercase"))
    use_emoji = bool(ov.get("emoji"))

    out = [_ASS_HEADER.format(**st)]

    for line in _chunk_words(words, max_words_per_line):
        parts: list[str] = []
        for w in line:
            cs = max(1, int(round((w["end"] - w["start"]) * 100)))
            txt = _clean(w["word"])
            if uppercase:
                txt = txt.upper()
            if use_emoji:
                key = txt.lower().strip(_EMOJI_STRIP)
                emoji = EMOJI_MAP.get(key)
                if emoji:
                    txt = f"{txt}{emoji}"
            parts.append(f"{{\\k{cs}}}{txt}")
        text = " ".join(parts)
        start = _ass_time(line[0]["start"])
        end = _ass_time(line[-1]["end"] + 0.15)  # small hold for readability
        out.append(_DIALOGUE.format(start=start, end=end, style=style, text=text))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    return path
