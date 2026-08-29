# WarriorIQ kickboxing dataset

The market-grade action classifier requires **real labeled kickboxing sequences**, not random/untrained weights.

## Dataset rule that cannot be compromised

Keep whole fights separated across train/validation/test. Clips from one source fight must never be randomly split into both training and validation because that inflates accuracy through leakage.

## Technique labels

The current temporal technique classes are:

- `none`
- `jab`
- `cross`
- `left_hook`, `right_hook`
- `left_uppercut`, `right_uppercut`
- `backfist`, `spinning_backfist`
- `left_round_kick`, `right_round_kick`
- `left_front_kick`, `right_front_kick`
- `left_push_kick`, `right_push_kick`
- `left_knee`, `right_knee`

Low/body/head kick are refined by the separate target/contact engine from the generic round-kick motion. Outcome labels also remain separate: `clean`, `likely_landed`, `blocked`, `checked`, `missed`, `uncertain`; target is `head`, `body`, or `leg`.

## Sequence format

Store each pose sequence in `dataset/sequences/` as an `.npz` with:

- `x`: float32 array shaped `(T, 102)`
- `y`: integer class index matching `core.temporal_model.ACTION_CLASSES`
- `fight_id`: strongly recommended string scalar identifying the original source fight

If `fight_id` is omitted, the trainer groups files using the filename prefix before `__`, e.g. `fight42__000123.npz`.

## Workflow

1. In the private development workspace, open a WarriorIQ result and label every proposal. Use **Not an action** for false detections, add actions the detector missed, and pause the video at exact contact to correct timing. This is model-development work, not a task required from normal WarriorIQ customers.
2. Use the **Accuracy** page to monitor fighter identity, technique, side, target, outcome and WAKO-legality agreement. No percentage is displayed until human corrections exist.
3. Collect at least 20 real actions and 20 negative labels across at least two different fights before the first experimental training run. This is only a minimum engineering gate, not proof of market readiness.
4. For a production-scale dataset, also annotate full kickboxing fights/sparring in Roboflow or another suitable video annotation workflow.
5. Build balanced labels across stance, camera angle, lighting, clothing, ruleset and fighter skill level.
6. Include negative motion: guard movement, feints, steps, bounces, clinch movement, referee motion and pre-round movement so these are not learned as strikes.
7. Extract the 102-dimensional pose+velocity sequences used by WarriorIQ.
8. Audit first with `python tools/audit_accuracy_dataset.py --require experimental`. Invalid shapes, non-finite values, duplicate sequences and fight/test leakage are excluded or rejected rather than silently counted.
9. Train with `python tools/train_temporal_model.py --dataset-version corrections-v1`.
10. The trainer performs a **fight-group validation split**, not a random clip split.
11. Keep another completely untouched full-fight test directory with all 17 supported temporal classes represented. Evaluate and promote only after it passes overall accuracy, every-class recall and every-class F1: `python tools/evaluate_temporal_model.py --test-data dataset/untouched_test --promote`.
12. A technique checkpoint alone cannot unlock fight facts. Run the separate end-to-end benchmark for fighter identity, target, outcome, ruleset legality and contact timing: `python tools/promote_end_to_end_validation.py --annotations private-ground-truth.json --promote`. The tool refuses promotion when any required dimension is missing or below its gate.

WarriorIQ should not be called market-best until complete unseen-fight benchmarks pass identity, technique, contact/outcome, target, scoring agreement and real-time performance without manual corrections after initial fighter selection.
