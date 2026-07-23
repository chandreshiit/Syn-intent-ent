# Results

Every number below is backed by a committed JSON under `results/`. Commands to
regenerate them are in [REPRODUCE.md](REPRODUCE.md).

## Headline

| Finding                                         | Evidence                                                                                                                                                                                                               |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Text is not the bottleneck for speech SLU       | Synthetic-text BERT scores **100.00%** intent accuracy on real Skit-S2I transcripts, including the 608-utterance partition whose transcripts never appear in training. The audio model on the same data scores 49.57%. |
| The speech gap is speaker identity, not channel | Matching the telephony channel to within ~7% on every acoustic metric moved transfer **+0.93pp**. Cloning the speakers' voices moved it **+22.29pp**.                                                                  |
| Confirmed against a real-trained control        | Synth-specific penalty is **24.7pp** on the speaker axis vs **1.3pp** on the channel axis — about 20x. A real-trained model shows a 7.3pp SNR effect too, so channel difficulty is intrinsic, not synthetic.           |
| The text gap is pragmatic register              | First-person framing **+4.59pp**, explanatory **+4.36pp**, colloquial **+2.55pp**, politeness **-0.05pp** (null). Vocabulary and slot coverage were not the bottleneck.                                                |
| Replicates across benchmarks and acoustics      | Skit-S2I at 8 kHz telephony **+22.29pp**; SNIPS at 16 kHz close-field **+7.93pp**.                                                                                                                                     |
| ~10x less annotation for near-ceiling           | SNIPS synthetic + 10% real (142 utterances) reaches 92.92% intent / 93.60% slot, against 96.88 / 96.12 for real-only.                                                                                                  |

## Skit-S2I: successive interventions (Table 7)

Synthetic-only intent accuracy on the real test set, 1,400 utterances.

| Intervention                       | Backbone | Synth-only | Cumulative gain |  Step gain |
| ---------------------------------- | -------- | ---------: | --------------: | ---------: |
| Baseline (generic voices, clean)   | tiny.en  |      49.57 |               — |          — |
| + channel matching (telephony)     | tiny.en  |      50.50 |           +0.93 |      +0.93 |
| + speaker identity (voice cloning) | tiny.en  |      72.79 |          +23.22 | **+22.29** |
| + larger backbone                  | base.en  |      76.21 |          +26.64 |      +3.43 |

Both columns are shown because the paper's Table 7 reports cumulative gain while
its Section 5.3 argument compares _step_ gains (channel +0.93 against voice
+22.29). The step gains are the ones that carry the causal claim.

`results/domain_tuning/skit_s2i_domain_tuning*.json`

The ordering matters to the argument. Channel matching came first because it was
the hypothesis we set out to confirm. It failed, and that failure is what
located the real cause.

## Identity vs channel, confound-controlled (Table 8)

| Axis                         | Synth-trained | Real-trained | Synth-specific |
| ---------------------------- | ------------: | -----------: | -------------: |
| Between-speaker error spread |        47.6pp |       22.9pp |     **24.7pp** |
| Within-speaker SNR effect    |         8.6pp |        7.3pp |      **1.3pp** |

`results/diagnostics/skit_s2i_identity_diagnostic{,_realonly_control}.json`

Three independent lines converge: intervention (~24x), raw observation (~5.5x),
and confound-controlled (~20x). The two most rigorous agree.

## SNIPS: single voice vs cloned voices (Table 9)

| Condition            | Single voice | Cloned (51 voices) |     Δ |
| -------------------- | -----------: | -----------------: | ----: |
| Synth-only (0% real) |        66.29 |          **74.22** | +7.93 |
| + 25% real           |        77.90 |              78.19 | +0.28 |
| + 100% real          |        86.97 |          **88.39** | +1.42 |
| Real-only (shared)   |        87.54 |              87.54 |     — |

`results/domain_tuning/snips_audio_transfer{,_f5cloned}.json`

This is the strongest form of the claim. Same dataset, text, model, split, and
seed — the _only_ variable changed is the synthetic voices, and it flips
synthetic data from harmful (-0.57pp below real-only) to helpful (+0.85pp above).

The 25% row barely moves, which is what the identity hypothesis predicts: once
real audio supplies speaker diversity, the cloning advantage washes out.

## Register-family ablation (Table 10)

Five seeds, equal utterance budget per family.

| Register family       | +utts | Intent acc (mean±std) |     Δ | Δ/utt |
| --------------------- | ----: | --------------------: | ----: | ----: |
| Baseline (cleaned v2) |     0 |          82.77 ± 0.33 |     — |     — |
| First-person state    |    48 |      **87.37 ± 0.58** | +4.59 | 0.096 |
| Explanatory / causal  |    60 |          87.14 ± 0.22 | +4.36 | 0.073 |
| Colloquial verbs      |    60 |          85.33 ± 0.21 | +2.55 | 0.043 |
| Indirect / politeness |    52 |          82.72 ± 0.57 | -0.05 |    ~0 |

`results/ablations/snips_family_ablation_results.json`

The two families that help are the ones where intent is conveyed through
_described state_ rather than an imperative verb. Politeness is lexically
adjacent to templates already present, and adds nothing.

## SNIPS real-slice sweeps (Table 11)

Snips NLU (LogReg intent + CRF slots), across three synthetic corpus versions.

| Real ratio | v1 int | v2 int |    v3 int | v1 slot | v2 slot | v3 slot |
| ---------: | -----: | -----: | --------: | ------: | ------: | ------: |
|         0% |  83.00 |  83.57 | **89.24** |   82.54 |   84.45 |   86.87 |
|         5% |  85.27 |  87.25 |     90.65 |   85.53 |   89.35 |   89.56 |
|        10% |  88.95 |  89.80 |     92.92 |   90.19 |   91.92 |   93.60 |
|        25% |  92.92 |  93.20 |     94.62 |   93.21 |   94.25 |   94.77 |
|        50% |  94.90 |  94.90 |     94.90 |   96.07 |   96.32 |   96.48 |
|       100% |  96.60 |  96.88 |     96.88 |   96.11 |   97.57 |   97.42 |
|  Real-only |  96.88 |        |           |   96.12 |         |         |

`results/domain_tuning/snips_domain_tuning_snipsnlu{,_v2,_v3}.json`

v1→v2 is data cleaning (+0.57pp intent — coverage was _not_ the bottleneck).
v2→v3 is register augmentation (+5.67pp intent, halving the gap to real-only
from 13.9pp to 7.6pp). Real-only is a single shared reference; it does not
depend on the synthetic version.

## MultiATIS++ per language (Table 6)

| Lang         | n train | mBERT intent | mBERT slot F1 | BiLSTM intent | BiLSTM slot F1 |
| ------------ | ------: | -----------: | ------------: | ------------: | -------------: |
| en           |   4,486 |       100.00 |         90.53 |        100.00 |          98.55 |
| es           |   4,486 |        99.55 |         82.21 |         98.99 |          83.48 |
| pt           |   4,486 |        99.44 |         84.42 |         98.54 |          86.31 |
| de           |   4,486 |        98.65 |         86.20 |         96.30 |          85.84 |
| fr           |   4,486 |        99.33 |         81.60 |         98.43 |          84.70 |
| zh           |   4,486 |        97.87 |         80.33 |         96.75 |          75.35 |
| ja           |   4,486 |        95.74 |         62.21 |         94.17 |          49.47 |
| **Mean (7)** |         |    **98.65** |     **83.93** |     **97.60** |      **80.53** |
| hi           |   1,440 |        94.13 |         69.13 |         87.81 |          64.93 |
| tr           |     578 |        88.81 |         57.35 |         72.73 |          44.90 |

`results/multiatis/multiatis_multilingual_baselines.json`

Hindi and Turkish use the benchmark's smaller low-resource splits. Their numbers
reflect a fixed BIO-tokenization bug: the tokenizer split on the Devanagari
virama (a Unicode Mn character outside `\w`) and mangled Turkish apostrophe
suffixes. Fixing it moved Hindi slot F1 +40.22pp and Turkish intent +36.50pp.

## Contamination (Table 5)

| Dataset     | Exact match | 4-gram Jaccard | 5-gram Jaccard | ROUGE-L | BLEU-4 |
| ----------- | ----------: | -------------: | -------------: | ------: | -----: |
| SNIPS       |       3.80% |          0.062 |          0.038 |   0.800 |   48.4 |
| Skit-S2I    |       1.20% |          0.067 |          0.044 |   0.601 |   26.0 |
| MultiATIS++ |       0.40% |          0.018 |          0.008 |   0.593 |   28.2 |

`results/contamination/summary.csv`

The SNIPS exact matches are structurally inevitable short template forms
("turn down the lights in the {room}"), not memorised benchmark text.

## Zero-shot LLM baseline

| Dataset     | Intent acc |           Slot F1 |
| ----------- | ---------: | ----------------: |
| SNIPS       |      90.37 |             0.915 |
| Skit-S2I    |      93.29 | n/a (intent-only) |
| MultiATIS++ |      47.85 |             0.403 |

`results/zero_shot_llm/`

Zero-shot prompting is strong on small ontologies and collapses on MultiATIS++
(18 intents, 84 slots) — which is exactly where a model fine-tuned on synthetic
data should be preferred.

## Negative results

Reported because they constrain the conclusion:

| Intervention                                   |         Effect |
| ---------------------------------------------- | -------------: |
| Telephony channel matching                     |        +0.93pp |
| Heavy SpecAugment + speed perturbation         |   **-11.93pp** |
| Light SpecAugment only                         |        -1.43pp |
| Multi-reference voice cloning (8 refs/speaker) |        -2.58pp |
| SNIPS vocabulary/slot cleanup alone            |        +0.57pp |
| Politeness/indirect templates                  | -0.05pp (null) |

Augmenting variation that the real test set does not contain actively hurts.
