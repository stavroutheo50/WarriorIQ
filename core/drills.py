"""Step-by-step versions of the coaching drills (owner, 2026-11-26).

The competitor teardown showed drills written as steps - set it up, do it,
what a clean rep looks like, reset - read far better than one line. These are
the five pose drills core/coaching.py prescribes (POSE_DIMENSIONS), keyed by
the same names, so a report shows the steps for the drill its own measurement
chose. Nothing here is measured: it is coaching instruction, kept generic
enough to be true for orthodox and southpaw fighters alike, and the stance
note says only what holds for both.
"""

from __future__ import annotations

DRILL_STEPS: dict[str, dict] = {
    "Guard-return audit": {
        "steps": [
            ("Set up", "A partner on pads or in light gloves at punching range. You in your fighting stance."),
            ("Do it", "Throw a two- or three-strike combination. On the last strike, bring both hands back to your face before you move."),
            ("Clean rep", "Chin down, rear hand at your cheek, lead hand at eye level, elbows in, all before your partner's tap arrives."),
            ("Reset", "One second after your last strike your partner taps a light counter at your head. If a hand was not home, the rep does not count: step out and go again."),
        ],
        "stance": "Whatever your stance, the hand that just punched comes back along the line it went out on, and the other hand never leaves your face.",
        "solo": "No partner today: 3 x 2 min of shadowboxing in front of a mirror or your phone camera, freezing for one count with both hands home after every combination.",
    },
    "Balanced-finish rounds": {
        "steps": [
            ("Set up", "A bag or a partner on pads. Start in stance with your weight centred between your feet."),
            ("Do it", "Throw your usual entry, a kick or a two- or three-strike combination, and finish it back in stance."),
            ("Clean rep", "You hold the finish for one count without hopping, leaning or crossing your feet, ready to throw again straight away."),
            ("Reset", "If you stumble or need an extra step to catch yourself, the rep does not count. Reset your stance and repeat it."),
        ],
        "stance": "After a kick, put the kicking leg back where it started unless you mean to switch stance; letting it fall forward is where balance goes.",
        "solo": "No partner today: 3 x 90 sec of shadow kicks and combinations, freezing for one count in stance after each.",
    },
    "Centre-line movement rounds": {
        "steps": [
            ("Set up", "Mark the centre of the ring or mat with tape. A partner on pads or in light sparring."),
            ("Do it", "After each exchange, leave at an angle, stepping off to the side rather than straight back, then work your way back to the mark."),
            ("Clean rep", "You never back straight up to the ropes, and you are back near the mark within two steps of every exchange."),
            ("Reset", "If your back touches the ropes or you retreat in a straight line, stop, walk back to the centre and restart."),
        ],
        "stance": "Whichever way you leave, make the first step with the foot on that side, so your feet never cross.",
        "solo": "No partner today: 3 x 2 min of shadowboxing around a marked spot, angling off after every combination and returning to it.",
    },
    "Forward-pressure rounds": {
        "steps": [
            ("Set up", "A partner on pads who keeps moving back and to the side, throwing the odd light counter."),
            ("Do it", "Every time your partner steps back, step in after them and take the space back, hands up."),
            ("Clean rep", "Lead foot first and rear foot following, feet staying at stance width: no lunging, no crossing, and you cut the angle instead of following them in a straight line."),
            ("Reset", "If you circle away or stand and wait, your partner calls reset and you start again from the centre."),
        ],
        "stance": "Moving forward, the lead foot steps first and the rear foot follows, in either stance.",
        "solo": "No partner today: 3 x 2 min of shadowboxing moving forward with short steps, a jab every two steps, never crossing your feet.",
    },
    "Step-count rounds": {
        "steps": [
            ("Set up", "A partner on pads calling combinations, or light sparring."),
            ("Do it", "Never throw more than two strikes from the same spot. After two, move: a step, a pivot or an angle."),
            ("Clean rep", "Your feet move before or after every pair of strikes, and you are never flat-footed for a third."),
            ("Reset", "Your partner counts. Three strikes without a step and you stop and restart."),
        ],
        "stance": "Change angles by pivoting on your lead foot, the front foot in either stance.",
        "solo": "No partner today: 4 x 2 min of shadowboxing with one rule: two strikes, then move.",
    },
}


def drill_steps(name: str) -> dict | None:
    """The step-by-step version of a drill, by its name (with or without "Fighter X · ")."""
    key = str(name or "").split(" · ", 1)[-1].strip()
    entry = DRILL_STEPS.get(key)
    if entry is None:
        return None
    return {
        "steps": [{"label": label, "text": text} for label, text in entry["steps"]],
        "stance": entry["stance"],
        "solo": entry["solo"],
    }
