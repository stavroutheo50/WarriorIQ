# External combat-sport data

These are source candidates, not proof that Warrior IQ can report a strike as a
fight fact. A published dataset license does not establish that its uploader
owned the underlying broadcast or participant rights. Keep source media and
derived models out of the production pipeline until those rights and an
untouched, in-domain fight evaluation are documented.

| Sport | Source | Available labels | Status for Warrior IQ |
| --- | --- | --- | --- |
| Taekwondo | [TKD-Kick3](https://doi.org/10.5281/zenodo.20390892), CC BY 4.0 derived pose data | Staged front, roundhouse, and axe kicks; no opponent contact or fight identity | Already imported into `sequences_tkd_kick3/` for offline research. The local audit found 326 valid sequences covering four sided classes, zero negative sequences, and no untouched fight test. Not train/release ready. Attribute the source if used. |
| Boxing | [BoxingVI](https://github.com/Bikudebug/BoxingVI) | Punch technique clips from YouTube sparring | Importer exists for feasibility experiments. No dataset license or underlying footage permission was verified; do not use for a commercial model. |
| Muay Thai | [RTM-FASS](https://github.com/asaravanabavan/RTM-FASS/tree/main/data) | Describes strike and fight annotations | Source videos, annotations, and processed sequences are not published with the repository. Its code license does not license absent data. |
| Muay Thai | [Muay ThAI](https://universe.roboflow.com/myspace-7h2qj/muay-thai) | 300 still images with broad object/action classes; CC BY 4.0 is listed | Candidate for isolated image experiments only, pending source-media rights and label audit. No temporal identity, impact, or outcome labels. Do not feed it to the temporal fight-action trainer. |
| MMA | [MMA Fight Analyzer](https://github.com/Maximilianb1/mma-fight-analyzer) | Five-second fight/non-fight, phase, and pressure labels from UFC broadcast clips | Academic research source, not commercial-clear fight footage; no individual strike/outcome labels. Do not import into the paid product. |
| MMA | [fight-judge](https://github.com/hasanfaesal/fight-judge) | Fighter boxes and poses from UFC images | Dataset is CC BY-NC-SA 4.0 and strike classification is planned, not provided. Do not use commercially. |
| MMA | [final_mma_set](https://universe.roboflow.com/initialfightdataset/final_mma_set) | 2,794 still images with 16 broad classes; CC BY 4.0 is listed | Candidate for isolated image experiments only, pending source-media rights and label audit. No time-accurate action or landed/blocked/missed evidence. Do not feed it to the temporal fight-action trainer. |

WAKO kickboxing rulesets can share movement examples, but each ruleset still
needs its own legality/scoring tests. MMA and Muay Thai also need their own
in-domain temporal data. Static images cannot prove contact, actor identity over
time, combinations, or scoring. No customer needs to label footage; without
pre-existing qualified labels, acquisition or independent annotation by a data
partner is the unresolved prerequisite for measured accuracy improvements.
