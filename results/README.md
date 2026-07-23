# Results index

Every number in the paper maps to a file here. All values below were verified
cell-by-cell against the paper before this repository was published.

## Paper table → file

| Paper table | File |
|---|---|
| 5 — Contamination | `contamination/summary.csv` (+ per-dataset `*_report.md`) |
| 6 — MultiATIS++ per language | `multiatis/multiatis_multilingual_baselines.json` |
| 7 — Skit-S2I interventions | four files, see below |
| 8 — Identity vs channel | `diagnostics/skit_s2i_identity_diagnostic{,_realonly_control}.json` |
| 9 — SNIPS single vs cloned voice | `domain_tuning/snips_audio_transfer{,_f5cloned}.json` |
| 10 — Register-family ablation | `ablations/snips_family_ablation_results.json` (+ `family_*.json`) |
| 11 — SNIPS real-slice sweeps | `domain_tuning/snips_domain_tuning_snipsnlu{,_v2,_v3}.json` |
| 12 — Skit-S2I real-slice sweep | same four files as Table 7 |
| Zero-shot LLM baseline | `zero_shot_llm/*_results.json` |
| 5-fold cross-validation | `cross_validation/*.json` |
| Cost-benefit | `cost_benefit/cost_benefit_table.md` |

## Table 7 / 12 — the four Skit-S2I audio conditions

All four runs share text, model, split, and seed. **Only the synthetic audio
differs**, which is what makes the comparison an intervention rather than a
correlation.

| Paper row | Synth-only | File |
|---|---:|---|
| Baseline (generic voices, clean) | 49.57 | `skit_s2i_domain_tuning_pre_telephony_v2.json` |
| + channel matching (telephony) | 50.50 | `skit_s2i_domain_tuning_parler_v3.json` |
| + speaker identity (voice cloning) | 72.79 | `skit_s2i_domain_tuning_f5_v1_single_ref.json` |
| + larger backbone (base.en) | 76.21 | `skit_s2i_domain_tuning_whisper_base.json` |

The filenames carry history rather than paper-row numbers, because they are what
the scripts actually emit. `pre_telephony_v2` is the clean-audio baseline;
`parler_v3` is that same audio after the telephony post-process.

## Negative results

Measured against the voice-cloned condition (72.79):

| File | Synth-only | Δ |
|---|---:|---:|
| `skit_s2i_domain_tuning_f5_v2_multi_ref.json` | 70.21 | −2.58 |
| `skit_s2i_domain_tuning_aug_b1_light_specaug.json` | 71.36 | −1.43 |
| `skit_s2i_domain_tuning_aug_synth_only.json` | 60.86 | −11.93 |

## Superseded runs, kept deliberately

Two files record earlier states rather than final results. They are kept because
the paper's argument refers to them.

- **`skit_s2i_domain_tuning_whisper_base_pre_real_only_retune.bak.json`** — the
  base.en run before the real-only arm was retuned. Its real-only value is 86.57
  against 92.64 in the final file. base.en was unstable at the shared schedule
  (lr 1e-4, 10 epochs), peaking mid-training and then degrading; retuning to
  lr 2e-5 / 20 epochs fixed it. We report the retuned value because it is the
  *stronger* real baseline, which makes the comparison against synthetic
  conservative rather than flattering.

- **`multiatis/multiatis_multilingual_baselines_pre_bio_fix.json`** — before the
  BIO-tokenization fix. The tokenizer split on the Devanagari virama (a Unicode
  Mn character outside `\w`) and mangled Turkish apostrophe suffixes. Fixing it
  moved Hindi slot F1 +40.22pp and Turkish intent +36.50pp. Keeping the pre-fix
  file makes that claim auditable.

## Known gap in a committed artifact

`diagnostics/skit_s2i_identity_diagnostic.json` (the synth-trained arm) predates
a later revision of the diagnostic script and **does not contain the
`within_speaker_snr_effect_pp` field**. The paper's 8.6pp figure for that cell of
Table 8 is therefore not directly readable from this file, though the other three
cells of that table are (47.6, 22.9, and 7.3 from the real-trained control).

Re-running the synth arm with the current script regenerates the field:

```bash
python experiments/diagnostics/skit_s2i_identity_diagnostic.py
```

## Reading the JSONs

Domain-tuning files share a schema:

```jsonc
{
  "per_ratio": [ { "ratio": 0.0, "test_intent_acc": 0.7279, "n_train": 10528 } ],
  "real_only_baseline": { "test_intent_acc": 0.9129 },
  "config": { "backbone": "tiny.en", "epochs": 10, "lr": 1e-4, "seed": 42 }
}
```

Accuracies are fractions, not percentages. The Snips NLU files use
`intent_accuracy` / `slot_micro_f1` instead of `test_intent_acc`.

Training logs are not committed: they are large, and they contain absolute local
paths. The JSONs carry every number the paper cites.
