from __future__ import annotations

import json
from html import escape
from pathlib import Path

from core.coaching import build_coaching, build_pose_coaching, build_training_plan
from core.config import SETTINGS
from core.evidence_trust import automated_evidence_trust
from core.scoring import event_legality, is_legal_event, is_verified_scoring_event, score_fight
from core.types import AnalysisRequest, DefenseEvent, RoundSpec, StrikeEvent


def _timeline_event_reliable(event: StrikeEvent) -> bool:
    """Keep only actions whose motion, anatomy and outcome all have evidence."""
    # A public "key moment" must be directly supported. Ambiguous likely
    # contacts stay out. A miss appears only when its distance evidence passes
    # the same strict action/contact/identity gates as landed moments.
    if event.outcome not in {"clean", "blocked", "checked", "missed"}:
        return False
    if float(event.confidence) < 0.84:
        return False
    if float(event.contact_confidence) < 0.86:
        return False
    conf = event.evidence.get("contact_attacker_conf") or event.evidence.get("peak_attacker_conf") or []
    endpoint = {"left_hand": 9, "right_hand": 10, "left_leg": 15, "right_leg": 16, "left_knee": 13, "right_knee": 14}.get(event.limb)
    if endpoint is None or len(conf) <= endpoint or float(conf[endpoint]) < 0.50:
        return False
    if event.family == "punch" and event.target == "leg":
        return False
    if float(event.metadata.get("attacker_identity_confidence", 1.0)) < 0.70:
        return False
    if float(event.metadata.get("opponent_identity_confidence", 1.0)) < 0.70:
        return False
    return True


def _identity_seed_safe(tracking: dict, fighter: str) -> bool:
    """Reject legacy/invalid detector seeds without distrusting manual anchors."""
    source_key = f"fighter_{fighter}_seed_source"
    iou_key = f"initial_iou_{fighter}"
    source = tracking.get(source_key)
    overlap = tracking.get(iou_key)
    # Older synthetic reports and tests predate seed diagnostics. Their caller
    # remains responsible for identity trust; real analyses now always record
    # both fields.
    if source is None and overlap is None:
        return True
    if source == "manual_anchor":
        return True
    return source == "pose_detector" and float(overlap or 0.0) >= SETTINGS.min_initial_iou


def refresh_identity_integrity(report: dict) -> dict:
    """Apply the current identity safety gate to new and legacy reports.

    This prevents an older high-coverage wrong-person track from remaining
    usable after the lock policy improves.  It also upgrades safe legacy
    reports with pose-only coaching when action labels are still unvalidated.
    """
    tracking = report.setdefault("tracking", {})
    identity_ready = {
        fighter: (
            _identity_seed_safe(tracking, fighter)
            and float(tracking.get(f"fighter_{fighter}_coverage", 0.0)) >= 0.45
        )
        for fighter in ("A", "B")
    }
    tracking["fighter_A_initial_lock_safe"] = identity_ready["A"]
    tracking["fighter_B_initial_lock_safe"] = identity_ready["B"]
    target = report.get("video", {}).get("analysis_target", "BOTH")
    required = ("A", "B") if target == "BOTH" else (target,)
    identity_safe = all(identity_ready.get(fighter, False) for fighter in required)
    integrity = report.setdefault("integrity", {})
    integrity["identity_evidence_trusted"] = identity_safe
    integrity["fighter_identity_trusted"] = identity_ready

    if not identity_safe:
        integrity["action_metrics_trusted"] = False
        integrity["coaching_evidence_mode"] = "withheld_identity_failure"
        failed = ", ".join(f"Fighter {fighter}" for fighter in required if not identity_ready.get(fighter, False))
        scorecard = report.setdefault("scorecard", {})
        scorecard.update({
            "available": False,
            "totals": {"A": None, "B": None},
            "rounds": [],
            "winner_estimate": None,
            "status": "identity_integrity_failed",
            "disclaimer": (
                f"Scorecard withheld because {failed} did not pass the fighter-identity gate. "
                "Return to fighter selection and analyze again; person coverage alone cannot prove identity."
            ),
        })
        report["key_moments"] = []
        report["illegal_moves"] = []
        metrics = report.get("metrics", {})
        for fighter in ("A", "B"):
            if identity_ready[fighter] and fighter in metrics:
                pose_coaching = build_pose_coaching(fighter, metrics[fighter])
                report.setdefault("coaching", {})[fighter] = pose_coaching
                report.setdefault("training_plan", {})[fighter] = build_training_plan(
                    pose_coaching, fighter, metrics[fighter]
                )
            elif not identity_ready[fighter]:
                report.setdefault("coaching", {})[fighter] = {
                    "strengths": [], "improvements": [], "drills": [],
                    "note": "Coaching withheld because this fighter did not pass the identity-integrity gate.",
                }
                report.setdefault("training_plan", {})[fighter] = []
        return report

    if not bool(integrity.get("action_metrics_trusted", False)):
        integrity["coaching_evidence_mode"] = "pose_only"
        metrics = report.get("metrics", {})
        for fighter in required:
            if fighter not in metrics:
                continue
            pose_coaching = build_pose_coaching(fighter, metrics[fighter])
            report.setdefault("coaching", {})[fighter] = pose_coaching
            report.setdefault("training_plan", {})[fighter] = build_training_plan(
                pose_coaching, fighter, metrics[fighter]
            )
    return report


def build_preliminary_scorecard(
    events: list[StrikeEvent],
    ruleset: str,
    round_numbers: list[int],
    tracking: dict,
    analysis_target: str,
) -> dict:
    """Build a visible, explicitly unvalidated estimate from action candidates.

    Tracking coverage can establish that both selected people were observed; it
    cannot validate the rule engine's action labels.  This score therefore has
    a separate status and vocabulary from validated or human-confirmed facts.
    """
    coverage_a = float(tracking.get("fighter_A_coverage", 0))
    coverage_b = float(tracking.get("fighter_B_coverage", 0))
    minimum_coverage = min(coverage_a, coverage_b)
    coverage_ok = minimum_coverage >= SETTINGS.min_tracking_coverage_for_score
    scorecard = score_fight(events, ruleset, round_numbers, [], reliable=coverage_ok)
    candidate_count = int(scorecard.get("verified_actions_counted", 0))
    scorecard["evidence"] = {
        "required_tracking_coverage_each": SETTINGS.min_tracking_coverage_for_score,
        "fighter_A_tracking_coverage": coverage_a,
        "fighter_B_tracking_coverage": coverage_b,
        "scoring_action_candidates": candidate_count,
        "duplicate_action_candidates_removed": int(scorecard.get("duplicate_action_candidates_removed", 0)),
        "action_confidence_required": 0.72,
        "contact_confidence_required": 0.62,
        "evidence_source": "unvalidated_action_candidates",
    }
    if analysis_target != "BOTH":
        scorecard.update({
            "available": False,
            "totals": {"A": None, "B": None},
            "rounds": [],
            "winner_estimate": None,
            "status": "both_fighters_required",
            "disclaimer": "To receive an estimated scorecard, choose Analyze both fighters. A one-fighter analysis does not count the opponent's points.",
        })
    elif not coverage_ok:
        scorecard["status"] = "insufficient_observation_coverage"
        scorecard["disclaimer"] = (
            f"A preliminary score requires at least {SETTINGS.min_tracking_coverage_for_score*100:.0f}% observation coverage "
            f"for both fighters. This analysis produced A {coverage_a*100:.1f}% and B {coverage_b*100:.1f}%."
        )
    elif candidate_count < 1:
        scorecard.update({
            "available": False,
            "totals": {"A": None, "B": None},
            "rounds": [],
            "winner_estimate": None,
            "status": "no_scoring_candidates",
            "disclaimer": "No preliminary score is shown because the automatic action engine found no scoring candidates with enough evidence.",
        })
    else:
        scorecard["available"] = True
        scorecard["status"] = "preliminary_unvalidated"
        scorecard["disclaimer"] = (
            "Preliminary computer estimate from high-confidence action candidates. The actions are not verified fight facts; "
            "the report does not ask you to label or correct them. This is not an official judges' score."
        )
    return scorecard


def build_report(
    req: AnalysisRequest,
    original_name: str,
    rounds: list[RoundSpec],
    events: list[StrikeEvent],
    defenses: list[DefenseEvent],
    metrics: dict,
    tracking: dict,
    performance: dict,
    classifier: dict,
) -> dict:
    evidence_trust = automated_evidence_trust(classifier)
    automated_evidence_trusted = bool(evidence_trust["automated_evidence_trusted"])
    tracking = dict(tracking)
    identity_ready = {
        fighter: (
            _identity_seed_safe(tracking, fighter)
            and float(tracking.get(f"fighter_{fighter}_coverage", 0.0)) >= 0.45
        )
        for fighter in ("A", "B")
    }
    tracking["fighter_A_initial_lock_safe"] = identity_ready["A"]
    tracking["fighter_B_initial_lock_safe"] = identity_ready["B"]
    required_fighters = ("A", "B") if req.analysis_target == "BOTH" else (req.analysis_target,)
    identity_evidence_trusted = all(identity_ready[fighter] for fighter in required_fighters)
    action_metrics_trusted = automated_evidence_trusted and identity_evidence_trusted
    round_numbers = [r.number for r in rounds if r.selected]
    minimum_coverage = min(float(tracking.get("fighter_A_coverage", 0)), float(tracking.get("fighter_B_coverage", 0)))
    scoring_reliable = action_metrics_trusted and minimum_coverage >= SETTINGS.min_tracking_coverage_for_score
    verified_events = [e for e in events if action_metrics_trusted and is_verified_scoring_event(e, req.ruleset)]
    key_events: list[StrikeEvent] = []
    timeline_candidates = [
        event for event in events
        if action_metrics_trusted and _timeline_event_reliable(event) and is_legal_event(event, req.ruleset)
    ]
    for event in sorted(timeline_candidates, key=lambda e: (-e.contact_confidence, -e.confidence, e.peak_time)):
        if all(abs(event.peak_time - kept.peak_time) >= 2.5 for kept in key_events):
            key_events.append(event)
        if len(key_events) >= 8:
            break
    key_events.sort(key=lambda e: e.peak_time)
    illegal_events: list[dict] = []
    for event in sorted(events, key=lambda item: (-item.confidence, item.peak_time)):
        if not action_metrics_trusted:
            break
        legal, reason = event_legality(event, req.ruleset)
        if legal or not _timeline_event_reliable(event):
            continue
        if any(event.fighter == kept["fighter"] and abs(event.peak_time - kept["peak_time"]) < 0.45 for kept in illegal_events):
            continue
        item = event.to_dict()
        item["legality_reason"] = reason
        illegal_events.append(item)
        if len(illegal_events) >= 12:
            break
    illegal_events.sort(key=lambda item: item["peak_time"])
    if action_metrics_trusted:
        scorecard = score_fight(events, req.ruleset, round_numbers, [], reliable=scoring_reliable)
        scorecard["evidence"] = {
            "required_tracking_coverage_each": SETTINGS.min_tracking_coverage_for_score,
            "fighter_A_tracking_coverage": float(tracking.get("fighter_A_coverage", 0)),
            "fighter_B_tracking_coverage": float(tracking.get("fighter_B_coverage", 0)),
            "verified_scoring_actions": int(scorecard.get("verified_actions_counted", len(verified_events))),
            "duplicate_action_candidates_removed": int(scorecard.get("duplicate_action_candidates_removed", 0)),
            "action_confidence_required": 0.72,
            "contact_confidence_required": 0.62,
            "evidence_source": "validated_model",
        }
    else:
        scorecard = build_preliminary_scorecard(events, req.ruleset, round_numbers, tracking, req.analysis_target)
        if req.analysis_target == "BOTH" and not identity_evidence_trusted:
            failed = ", ".join(f"Fighter {fighter}" for fighter in ("A", "B") if not identity_ready[fighter])
            scorecard.update({
                "available": False,
                "totals": {"A": None, "B": None},
                "rounds": [],
                "winner_estimate": None,
                "status": "identity_integrity_failed",
                "disclaimer": (
                    f"Scorecard withheld because {failed} did not pass the fighter-identity gate. "
                    "Return to fighter selection and analyze again; person coverage alone cannot prove identity."
                ),
            })
    if req.analysis_target != "BOTH":
        scorecard["available"] = False
        scorecard["totals"] = {"A": None, "B": None}
        scorecard["rounds"] = []
        scorecard["winner_estimate"] = None
        scorecard["disclaimer"] = "To receive an estimated scorecard, choose Analyze both fighters. A one-fighter analysis does not count the opponent's points."
    elif action_metrics_trusted and not scoring_reliable:
        scorecard["disclaimer"] = (
            f"An estimated score requires at least {SETTINGS.min_tracking_coverage_for_score*100:.0f}% verified tracking "
            f"for both fighters. This analysis produced A {tracking.get('fighter_A_coverage', 0)*100:.1f}% and "
            f"B {tracking.get('fighter_B_coverage', 0)*100:.1f}%."
        )
    insufficient_coaching = {
        "strengths": [], "improvements": [], "drills": [],
        "note": "WarriorIQ did not invent coaching advice from unverified action candidates.",
    }
    # Validated actions support technique coaching. Until that release gate is
    # passed, identity-safe pose measurements still support useful movement
    # work without presenting candidate strikes as facts.
    coaching: dict[str, dict] = {}
    for fighter in ("A", "B"):
        if action_metrics_trusted and identity_ready[fighter]:
            coaching[fighter] = build_coaching(fighter, metrics, events)
        elif identity_ready[fighter]:
            coaching[fighter] = build_pose_coaching(fighter, metrics[fighter])
        else:
            coaching[fighter] = dict(insufficient_coaching)
            coaching[fighter]["note"] = "Coaching withheld because this fighter did not pass the identity-integrity gate."
    coaching_a, coaching_b = coaching["A"], coaching["B"]

    report = {
        "product": {"name": "WarriorIQ", "version": "1.0"},
        "video": {
            "label": "Fight analysis",
            "fight_type": req.fight_type,
            "analysis_target": req.analysis_target,
            "focus_fighter": req.focus_fighter or req.analysis_target,
        },
        "setup": {
            "ruleset": req.ruleset,
            "round_count": req.round_count,
            "round_duration_seconds": req.round_duration_seconds,
            "break_duration_seconds": req.break_duration_seconds,
            "selected_rounds": req.selected_rounds,
            "start_seconds": req.start_seconds,
            "end_seconds": req.end_seconds,
            "fighter_a_box": req.fighter_a_box,
            "fighter_b_box": req.fighter_b_box,
        },
        "rounds": [
            {
                "number": r.number,
                "start_seconds": r.start_seconds,
                "end_seconds": r.end_seconds,
                "selected": r.selected,
            }
            for r in rounds
        ],
        "performance": performance,
        "tracking": tracking,
        "classifier": classifier,
        "metrics": metrics,
        "scorecard": scorecard,
        "events": [e.to_dict() for e in events],
        "key_moments": [e.to_dict() for e in key_events],
        "illegal_moves": illegal_events,
        "defenses": [d.to_dict() for d in defenses],
        "coaching": {"A": coaching_a, "B": coaching_b},
        "training_plan": {
            "A": build_training_plan(coaching_a, "A", metrics["A"]),
            "B": build_training_plan(coaching_b, "B", metrics["B"]),
        },
        "integrity": {
            **evidence_trust,
            "identity_evidence_trusted": identity_evidence_trusted,
            "fighter_identity_trusted": identity_ready,
            "action_metrics_trusted": action_metrics_trusted,
            "coaching_evidence_mode": "validated_actions" if action_metrics_trusted else "pose_only" if identity_evidence_trusted else "withheld_identity_failure",
            "human_review_complete": False,
            "no_demo_statistics": True,
            "uncertainty_policy": "WarriorIQ leaves a fighter/action unavailable or uncertain when evidence is insufficient rather than inventing a result.",
            "scoring_status": scorecard["disclaimer"],
            "minimum_fighter_coverage_for_score": SETTINGS.min_tracking_coverage_for_score,
            "model_validation_status": "A custom WarriorIQ temporal checkpoint is used only when present. Otherwise the multi-frame deterministic classifier is labeled as fallback.",
            "rules_reference": "WAKO Rules revision 25.10.2022; K-1 2026 amendment takes effect 01.01.2027 and is not applied before that date.",
        },
    }
    return report


def write_report(job_dir: Path, report: dict) -> tuple[Path, Path]:
    job_dir.mkdir(parents=True, exist_ok=True)
    json_path = job_dir / "report.json"
    html_path = job_dir / "report.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fmt(value):
        if value is None:
            return "Unavailable"
        if isinstance(value, float):
            return f"{value:.3f}"
        return escape(str(value))

    def fighter_card(name: str) -> str:
        m = report["metrics"][name]
        attacks = m["attacks"]
        accuracy = "Unavailable" if attacks["accuracy"] is None else f"{attacks['accuracy']*100:.1f}%"
        return f"""
        <section class='card'>
          <h2>Fighter {name}</h2>
          <div class='big'>{attacks['landed']} / {attacks['attempts']}</div>
          <div class='muted'>clean/likely landed / detected attempts</div>
          <table>
            <tr><td>Accuracy</td><td>{accuracy}</td></tr>
            <tr><td>Pose coverage</td><td>{m['pose_coverage']*100:.1f}%</td></tr>
            <tr><td>Strongest weapon</td><td>{fmt(m['strongest_weapon'])}</td></tr>
            <tr><td>Combinations</td><td>{m['combinations']['count']}</td></tr>
            <tr><td>Counters</td><td>{m['counters']['count']}</td></tr>
          </table>
        </section>
        """

    event_rows = "".join(
        f"<tr><td>{e['round_number'] or '-'}</td><td>{e['peak_time']:.2f}</td><td>{escape(e['fighter'])}</td>"
        f"<td>{escape(e['technique'].replace('_',' '))}</td><td>{escape(e['outcome'])}</td><td>{escape(str(e['target']))}</td></tr>"
        for e in report["key_moments"]
    )

    coaching_html = ""
    for fighter in ("A", "B"):
        c = report["coaching"][fighter]
        coaching_html += f"<section class='card'><h2>Coaching · Fighter {fighter}</h2>"
        coaching_html += "<h3>Strengths</h3><ul>" + "".join(
            f"<li><strong>{escape(x['title'])}</strong> — {escape(x['detail'])}</li>" for x in c["strengths"]
        ) + "</ul>"
        coaching_html += "<h3>Improvements</h3><ul>" + "".join(
            f"<li><strong>{escape(x['title'])}</strong> — {escape(x['detail'])}</li>" for x in c["improvements"]
        ) + "</ul></section>"

    score = report["scorecard"]
    scorecard_html = (
        f"<p class='big'>Fighter A {score['totals']['A']} · Fighter B {score['totals']['B']}</p>"
        if score.get("available") else
        f"<p>{escape(score['disclaimer'])}</p>"
    )

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>WarriorIQ Report</title>
<style>
body{{font-family:Inter,Arial,sans-serif;background:#090c12;color:#eef2f7;max-width:1150px;margin:0 auto;padding:32px}}
header{{display:flex;justify-content:space-between;align-items:end;margin-bottom:24px}}h1{{font-size:38px;margin:0}}.muted{{color:#8b98aa}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.card{{background:#121824;border:1px solid #283346;border-radius:18px;padding:20px;margin-bottom:18px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #263144;text-align:left}}.big{{font-size:32px;font-weight:800}}.pill{{background:#1d2838;padding:7px 10px;border-radius:999px}}a{{color:#8cc8ff}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div><h1>WarriorIQ</h1><div class='muted'>Fight analysis</div></div><div class='pill'>{escape(report['scorecard']['ruleset_label'])}</div></header>
<div class='grid'>{fighter_card('A')}{fighter_card('B')}</div>
<section class='card'><h2>Performance</h2><p>Segment: {report['performance']['segment_duration_seconds']:.1f}s · Analysis: {report['performance']['analysis_seconds']:.1f}s · Speed: {report['performance']['realtime_speed']:.2f}× realtime · Within budget: {report['performance']['within_video_length_budget']}</p></section>
<section class='card'><h2>Estimated scorecard</h2>{scorecard_html}</section>
<section class='card'><h2>Evidence timeline</h2><table><thead><tr><th>Round</th><th>Time</th><th>Fighter</th><th>Technique</th><th>Outcome</th><th>Target</th></tr></thead><tbody>{event_rows}</tbody></table></section>
{coaching_html}
<section class='card'><h2>Integrity</h2><p>{escape(report['integrity']['uncertainty_policy'])}</p><p>{escape(report['integrity']['scoring_status'])}</p></section>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return json_path, html_path
