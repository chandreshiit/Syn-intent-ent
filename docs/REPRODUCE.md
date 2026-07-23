# Reproducing the paper

Every table in the paper maps to one command here. Result JSONs for all of them
are already committed under `results/`, so you can check any number without
running anything.

Before starting: install the environments ([README](../README.md#installation))
and stage the corpora ([DATA.md](DATA.md)). All commands run from the repository
root.

## Cost

| Tier      | Runtime      | What                                                 |
| --------- | ------------ | ---------------------------------------------------- |
| Free      | seconds      | inspect committed JSONs under `results/`             |
| Cheap     | minutes, CPU | contamination check, SNIPS text generation           |
| Moderate  | 1-3 h, GPU   | register ablation, SNIPS sweeps                      |
| Expensive | 10-20 h, GPU | Skit-S2I audio regeneration, MultiATIS++ 9 languages |

Start with the cheap tier. It exercises the install end to end and reproduces a
published table exactly.

## Verify the install (no GPU, no corpora)

```bash
python -c "from slu_gap.models import bert_alone, lstm_alone; print('ok')"
python -m compileall -q src experiments pipelines && echo "ok"
```

## Table 5 — Contamination

The cheapest check that reproduces a published number exactly.

```bash
python experiments/contamination/contamination_check.py
```

Expected exact-match rates: SNIPS 3.80%, Skit-S2I 1.20%, MultiATIS++ 0.40%.
Committed at `results/contamination/summary.csv`.

## Table 6 — MultiATIS++ per language

Nine languages x two architectures. Run per language; training all nine in one
process fragments CUDA memory and OOMs on the later ones.

```bash
python experiments/multiatis/multiatis_bert_per_lang.py --epochs 20
python experiments/multiatis/multiatis_multilingual_baselines.py \
    --languages en --models bert lstm --epochs 20
```

Committed at `results/multiatis/multiatis_multilingual_baselines.json`.

`results/multiatis/multiatis_multilingual_baselines_pre_bio_fix.json` holds the
pre-fix numbers. The difference is the BIO-tokenization fix described in
`pipelines/multiatis/fix_bio_tokenization.py`, worth +40.22pp slot F1 on Hindi
and +39.74pp on Turkish — the tokenizer was splitting on the Devanagari virama.

## Tables 7, 12 — Skit-S2I interventions and real-slice sweep

One script, four audio conditions. Only `--synth-csv` / `--synth-base` change,
which is what makes the comparison a controlled intervention.

```bash
# generic voices, clean (baseline)      -> 49.57
# + telephony channel matching          -> 50.50
# + F5 voice cloning                    -> 72.79
# + base.en backbone                    -> 76.21
python experiments/domain_tuning/skit_s2i_domain_tuning.py \
    --ratios 0 0.25 1.0 --epochs 10 --lr 1e-4 --seed 42
```

Committed as `results/domain_tuning/skit_s2i_domain_tuning*.json`, one file per
condition. `_parler_v3` is telephony-matched, `_f5_v1_single_ref` is voice-cloned,
`_whisper_base` is the larger backbone.

To regenerate the cloned audio (expensive, needs the `f5tts` environment):

```bash
python pipelines/voice_cloning/skit_s2i/pick_references.py
python pipelines/voice_cloning/skit_s2i/generate_full.py
```

## Table 8 — Identity vs channel, with real-trained control

The control arm is what makes this causal rather than correlational: a
real-trained model shows a 7.3pp SNR effect too, so only ~1.3pp of the channel
effect is attributable to synthetic training, against ~24.7pp for speaker identity.

```bash
python experiments/diagnostics/skit_s2i_identity_diagnostic.py
python experiments/diagnostics/skit_s2i_identity_diagnostic.py --train-source real
```

Committed at `results/diagnostics/skit_s2i_identity_diagnostic{,_realonly_control}.json`.

## Table 9 — SNIPS transfer, single voice vs cloned

The within-dataset causal flip. Only the synthetic voices change; synth+100%real
moves from 0.57pp _below_ real-only to 0.85pp _above_ it.

```bash
python experiments/domain_tuning/snips_audio_transfer.py                    # MMS, 1 voice
python experiments/domain_tuning/snips_audio_transfer.py --synth-audio f5   # F5, 51 voices
```

Committed at `results/domain_tuning/snips_audio_transfer{,_f5cloned}.json`.

To regenerate the cloned audio — note that references are drawn only from the
real _train_ split, so no test-set audio leaks into training:

```bash
python pipelines/voice_cloning/snips/pick_references_snips.py
python pipelines/voice_cloning/snips/generate_snips_f5.py
```

## Table 10 — Register-family ablation

Five seeds per family at an equal utterance budget, so families are compared at
matched cost rather than matched effort.

```bash
python experiments/ablations/snips_synth_family_ablation.py --seeds 5
```

Committed at `results/ablations/snips_family_ablation_results.json` plus one
`family_*.json` per condition.

The baseline standard deviation is 0.33pp, so only gains above ~0.7pp (2 sigma)
are real. This is why indirect/politeness at -0.05pp is reported as a null
result rather than a small negative one.

## Table 11 — SNIPS real-slice sweeps

Needs the `snipsnlu` environment (Python 3.8, WSL on Windows).

```bash
python experiments/data_prep/prepare_synth_snips_for_snipsnlu.py
python experiments/domain_tuning/snips_domain_tuning_snipsnlu.py           # v1
python experiments/domain_tuning/snips_domain_tuning_snipsnlu.py --version v2
python experiments/domain_tuning/snips_domain_tuning_snipsnlu.py --version v3
```

Committed at `results/domain_tuning/snips_domain_tuning_snipsnlu{,_v2,_v3}.json`.

The v1→v2→v3 progression is the diagnostic loop in action: v2 is data cleaning,
v3 adds register-targeted templates chosen from the v2 error analysis. Synth-only
intent accuracy goes 83.00 → 83.57 → 89.24.

The JointBERT arm of the same sweep:

```bash
python experiments/domain_tuning/snips_domain_tuning.py --epochs 20
```

## Supporting experiments

```bash
# Zero-shot LLM baseline (needs Ollama)
python experiments/zero_shot_llm/zero_shot_baseline.py

# 5-fold CV -- WITHIN-distribution, not transfer. See the caveat below.
python experiments/cross_validation/snips_5fold_jointbert.py --epochs 20
python experiments/cross_validation/snips_5fold_whisper.py --epochs 20
python experiments/cross_validation/snips_5fold_snipsnlu.py

# Text is not the bottleneck: 100% intent accuracy on real transcripts
python experiments/diagnostics/skit_s2i_text_only_baseline.py

# The error analysis that told us which registers to add
python experiments/diagnostics/snips_synthonly_error_analysis.py

# Negative result: training-side augmentation HURTS synth-to-real transfer
python experiments/ablations/skit_s2i_domain_tuning_aug.py
```

> **On the cross-validation numbers.** Synthetic-to-synthetic 5-fold CV reaches
> 99.7% on SNIPS audio, but true synthetic-to-real transfer is 66.3% — a 33pp
> drop the CV completely hides. The CV results are reported for the standard
> deviations they provide, not as evidence that synthetic audio works. If you
> take one methodological point from this repository, take that one.

## Regenerating synthetic corpora

```bash
# SNIPS: template generation, EN->FR translation, slot projection, TTS  (~1 h)
python pipelines/snips/00_generate_source_commands.py
python pipelines/snips/01_translate_to_multilingual.py
python pipelines/snips/02_project_slots_crosslingual.py
python pipelines/snips/03_synthesize_multilingual_speech.py
python pipelines/snips/04_process_multilingual_dataset.py

# Skit-S2I: LLM seed-pattern generation, then TTS + telephony      (~11 h)
python pipelines/skit_s2i/generate_monolingual_SISE_data.py
python pipelines/skit_s2i/synthesize_speech.py

# MultiATIS++: generation, translation, BIO projection, TTS        (~17 h)
python pipelines/multiatis/00_generate_source_commands.py
# ... 01 through 04, then:
python pipelines/multiatis/fix_bio_tokenization.py
```

Step 00 for SNIPS should print 1,765 entries, 596 multi-slot, 0 BIO errors —
matching the real benchmark's per-intent counts exactly.
