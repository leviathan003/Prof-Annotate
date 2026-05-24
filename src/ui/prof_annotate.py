"""
bytemark/ui/prof_annotate.py
The arcane mascot of ByteMark — Prof. Annotate.

Pixel-art portrait + voice lines + colour tokens + presence manager
for every surface Prof. inhabits (splash, tutorial, alerts, his workshop).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

# ── Colour tokens (arcane palette) ───────────────────────────────────────────
PROF_GOLD = "#D4AF37"
PROF_GOLD_DIM = "#8A7128"
PROF_VIOLET = "#9D4EDD"
PROF_VIOLET_DEEP = "#3A1A55"
PROF_TEAL = "#00CFA0"
PROF_PARCHMENT = "#E8D5A0"
PROF_INK = "#1A0F22"
PROF_SHADOW = "#08040C"

# ── ASCII pixel-art portrait ─────────────────────────────────────────────────
# Rendered in a monospace label. Lines kept identical length (29 cols) so the
# figure stays aligned in every QFontMetrics environment.
PROF_PORTRAIT = r"""
              ✦
             ╱╲
            ╱✧ ╲
           ╱ ⟁  ╲
          ╱  ☾   ╲
         ╱ ✦   ✦  ╲
        ╱  ◆ ◇ ◆   ╲
       ╱▁▁▁▁▁▁▁▁▁▁▁▁╲
      ╱▒▒▒▒▒▒▒▒▒▒▒▒▒▒╲
     ▐░░╲▔▔▔▔▔▔▔▔▔▔╱░░▌
     ▐░░░╲          ╱░░▌
     ▐░░░ ◉   ⟁   ◉ ░░░▌
     ▐░░░  ╲      ╱  ░░▌
     ▐░░░   ╶◊◊╴     ░░▌
     ▐░░░  ╲▂▂▂▂▂╱   ░░▌
      ╲░░  ╲▒▒▒▒▒╱    ░╱
       ╲▔▔▔▔▔▔▔▔▔▔▔▔▔╱
        ▟█▙   ▼   ▟█▙
       ▟███▙ ▟█▙ ▟███▙
      ▟█ ⚜ █████████ ⚜ █▙
     ▟███ ◈   ⚡⚡⚡   ◈ ███▙
     ▌███   ━━━━━━━━━   ███▐
     ▌████   ✧ ◇ ✧    ████▐
     ▌█████  ⚜⚜⚜⚜⚜   █████▐
     ▌███████████████████▐
      ╲▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓╱
       ╱╲              ╱╲
      ╱  ╲            ╱  ╲
     ╱    ╲          ╱    ╲
""".strip("\n")


# A compact 10-line variant for tight surfaces (tutorial step popups,
# the workshop panel). Same character set as the full portrait.
PROF_PORTRAIT_SMALL = r"""
       ✦
      ╱╲
     ╱✧ ╲
    ╱ ⟁  ╲
   ╱  ☾   ╲
  ╱▁▁▁▁▁▁▁▁╲
  ▐ ◉ ⟁ ◉ ▌
  ▐  ╶◊╴   ▌
  ▐ ╲▂▂▂╱  ▌
 ▟███████████▙
 ▌◈  ⚡⚡⚡  ◈▐
 ▌  ⚜━━━⚜   ▐
""".strip("\n")


# A 6-line tiny variant for very tight surfaces — small laptops, tablets,
# or a workshop section that's been squashed by an aggressive splitter.
PROF_PORTRAIT_TINY = r"""
  ╱▔╲
 ╱⟁  ╲
 ◉ ◉
 ╲▽╱
▟███▙
⚡◈⚡
""".strip("\n")


# ── Prof.'s voice ────────────────────────────────────────────────────────────
# Reusable lines for the rest of the app to draw on. Tone: arcane, wise,
# slightly theatrical. Always addresses the user as "Annotator".

GREETINGS: list[str] = [
    "Ah, Annotator — you return to the workshop.",
    "The runes are ready, Annotator. Shall we begin?",
    "I sensed your arrival, Annotator. The dataset awaits.",
]

FAREWELLS: list[str] = [
    "Until the next annotation, Annotator.",
    "The labels are sealed. Rest well, Annotator.",
]

SPLASH_LINES: list[str] = [
    "✦ GPL V2 ✦",
    "An arcane workbench for those who tame raw data.",
    "I am Prof. Annotate — your guide through the labelling craft.",
]

TUTORIAL_INTRO = (
    "Welcome, Annotator. I am Prof. Annotate — keeper of bounding "
    "boxes, weaver of keypoints, and conjurer of masks.\n\n"
    "Shall I walk you through the workshop? Skip if you already "
    "know the craft."
)

TUTORIAL_OUTRO = (
    "The tour is concluded, Annotator. The dataset is yours to shape.\n\n"
    "Should you ever forget a binding, the Keybindings tome stands "
    "ready in the top bar."
)


# ── Animation frames for Prof.'s workshop ────────────────────────────────────
# Four subtly different portraits cycled in the workshop section so Prof.
# appears alive — blinking, focusing, casting. Every line is the same width.

PROF_FRAMES: list[str] = [
    # Frame A — observing (eyes open, calm)
    r"""
       ✦
      ╱╲
     ╱✧ ╲
    ╱ ⟁  ╲
   ╱  ☾   ╲
  ╱▁▁▁▁▁▁▁▁╲
  ▐ ◉ ⟁ ◉ ▌
  ▐  ╶◊╴   ▌
  ▐ ╲▂▂▂╱  ▌
 ▟███████████▙
 ▌◈  ⚡⚡⚡  ◈▐
 ▌  ⚜━━━⚜   ▐
""".strip("\n"),
    # Frame B — focused (eyes narrowed, drawing)
    r"""
       ✦
      ╱╲
     ╱✧ ╲
    ╱ ⟁  ╲
   ╱  ☾   ╲
  ╱▁▁▁▁▁▁▁▁╲
  ▐ ◎ ⟁ ◎ ▌
  ▐  ╶◇╴   ▌
  ▐ ╲▽▽▽╱  ▌
 ▟███████████▙
 ▌✦  ⚡✧⚡  ✦▐
 ▌  ⚜━━━⚜   ▐
""".strip("\n"),
    # Frame C — casting (stars flare, sigils bright)
    r"""
       ✧
      ╱╲
     ╱✦ ╲
    ╱ ⟁  ╲
   ╱  ☆   ╲
  ╱▁▁▁▁▁▁▁▁╲
  ▐ ◉ ✦ ◉ ▌
  ▐  ╶◈╴   ▌
  ▐ ╲▂▂▂╱  ▌
 ▟███████████▙
 ▌⚡  ✧✧✧  ⚡▐
 ▌  ⚜━━━⚜   ▐
""".strip("\n"),
    # Frame D — meditating (eyes closed, silent)
    r"""
       ·
      ╱╲
     ╱◇ ╲
    ╱ ⟁  ╲
   ╱  ☾   ╲
  ╱▁▁▁▁▁▁▁▁╲
  ▐ ‾ ⟁ ‾ ▌
  ▐  ╶◊╴   ▌
  ▐ ╲▂▂▂╱  ▌
 ▟███████████▙
 ▌◇  ◈◈◈  ◇▐
 ▌  ⚜━━━⚜   ▐
""".strip("\n"),
]


# Tiny animation frames — used when the workshop / popup column is short.
PROF_FRAMES_TINY: list[str] = [
    # A — observing
    r"""
  ╱▔╲
 ╱⟁  ╲
 ◉ ◉
 ╲▽╱
▟███▙
⚡◈⚡
""".strip("\n"),
    # B — focused
    r"""
  ╱▔╲
 ╱⟁  ╲
 ◎ ◎
 ╲▽╱
▟███▙
✦◈✦
""".strip("\n"),
    # C — casting
    r"""
  ╱▔╲
 ╱✦  ╲
 ◉ ◉
 ╲▽╱
▟███▙
⚡✧⚡
""".strip("\n"),
    # D — meditating
    r"""
  ╱▔╲
 ╱⟁  ╲
 ‾ ‾
 ╲▽╱
▟███▙
◈◇◈
""".strip("\n"),
]

# Status lines Prof. cycles through while he "works".
PROF_TASKS: list[str] = [
    "Tracing keypoints…",
    "Drawing wards of containment…",
    "Sealing the segmentation mask…",
    "Whispering to the data.yaml…",
    "Aligning the skeleton…",
    "Polishing label files…",
    "Consulting ancient annotations…",
    "Reading between the pixels…",
    "Charting class boundaries…",
    "Inscribing the keypoints…",
]

# Glyphs scrolled across the work-feed to convey arcane activity.
PROF_GLYPHS: str = "◆◇◈⚡✦✧⟁◉⊙⊕⊗⊘∴∵·"


# ── Presence manager ─────────────────────────────────────────────────────────
# Tracks whether Prof. is currently in his workshop section or has been
# "summoned" by a popup. Reference-counted so nested dialogs are safe.


class _ProfPresence(QObject):
    """Singleton coordinator. Emits `visibility_changed(True)` when Prof.
    returns to the workshop and `visibility_changed(False)` when he is
    summoned by a popup."""

    visibility_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._away_count = 0

    def summon(self) -> None:
        """A popup is taking Prof. — hide him from the workshop."""
        was_present = self._away_count == 0
        self._away_count += 1
        if was_present:
            self.visibility_changed.emit(False)

    def release(self) -> None:
        """The popup is done — Prof. may return to the workshop."""
        if self._away_count == 0:
            return
        self._away_count -= 1
        if self._away_count == 0:
            self.visibility_changed.emit(True)

    def force_reset(self) -> None:
        """Coerce Prof. back to the workshop unconditionally. Last-resort
        recovery if presence ever gets stuck (should not normally be needed —
        the dialog watcher uses three independent release triggers)."""
        was_absent = self._away_count > 0
        self._away_count = 0
        if was_absent:
            self.visibility_changed.emit(True)

    @property
    def is_in_workshop(self) -> bool:
        return self._away_count == 0


_PRESENCE = _ProfPresence()


def presence() -> _ProfPresence:
    """Return the module-level singleton."""
    return _PRESENCE
