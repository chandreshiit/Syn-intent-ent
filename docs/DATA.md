# Data

**No corpus is redistributed in this repository.** All three benchmarks carry
their own licences, one of which forbids redistribution outright. This document
explains how to obtain each one, where to stage it, and what you may do with it.

## Layout

Every path in the codebase resolves through `slu_gap.paths`, which reads:

| Variable          | Default            | Holds                           |
| ----------------- | ------------------ | ------------------------------- |
| `SLU_GAP_DATA`    | `<repo>/data`      | corpora and generated audio     |
| `SLU_GAP_RESULTS` | `<repo>/results`   | experiment result JSONs         |
| `SLU_GAP_SCRATCH` | system temp        | training checkpoints (multi-GB) |
| `SLU_GAP_MODELS`  | `<scratch>/models` | downloaded model weights        |

Point `SLU_GAP_SCRATCH` and `SLU_GAP_MODELS` at a volume with room to spare. The
speech sweeps write a full Whisper checkpoint per configuration, and a complete
run of every experiment will move tens of gigabytes through scratch.

Expected layout under `$SLU_GAP_DATA`:

```
snips_real_close/                 real SNIPS smart-lights, close-field
snips_multilingual_pipeline/      synthetic SNIPS (text, BIO, audio)
snips_f5_cloned_audio/            synthetic SNIPS audio, F5 voice-cloned
snips_synth_for_snipsnlu{,_v2,_v3}/   synthetic SNIPS in Snips NLU format
skit_s2i_real_audio/              real Skit-S2I audio + metadata.csv
skit_s2i_synthesis_pipeline/      synthetic Skit-S2I (text + generated audio)
multiatis_multilingual_pipeline/  synthetic MultiATIS++, 9 languages
```

## The three benchmarks

### SNIPS SLU (Smart Lights)

Released by Snips with the _Spoken Language Understanding for Voice Assistants_
work. Download `snips_slu_data_v1.0` from the Snips SLU benchmark release and
place `smart-lights-en-close-field/` at `$SLU_GAP_DATA/snips_real_close/`.

Then build the audio index the loaders expect:

```bash
python experiments/data_prep/prepare_real_snips.py
```

This writes `audio_index.json`, mapping each BIO row to its wav. Without it,
`slu_gap.datasets.load_real_snips_close` cannot align labels to audio.

### Skit-S2I

<https://github.com/skit-ai/speech-to-intent-dataset> . Also on the HuggingFace
Hub as `skit-ai/skit-s2i`.

**Licence: CC BY-NC 4.0.** Non-commercial use only, attribution required. You
may use and adapt it for research, but commercial use is not permitted, and any
derivative you redistribute must carry the same restriction. This is why the
real audio is not mirrored here.

Stage it, then extract the flat audio layout the experiments use:

```bash
python experiments/data_prep/extract_skit_s2i_real_audio.py
```

Cite:

> Nethil, K., Anandan, K., and Senani, U. _Speech to Intent Dataset_, 2022.

### MultiATIS++

**Requires a licence from the Linguistic Data Consortium: [LDC2021T04](https://catalog.ldc.upenn.edu/LDC2021T04).**

This is a hard constraint. The corpus extends ATIS
(LDC93S5, LDC94S19, LDC95S26), and **the data cannot be redistributed**. The
Apache-2.0 licence on Amazon's `multiatis` repository covers their _code_.

Obtain a copy through the LDC, then stage it under
`$SLU_GAP_DATA/multiatis_multilingual_pipeline/`.

Cite:

> Xu, W., Haider, B., and Mansour, S. _End-to-End Slot Alignment and Recognition
> for Cross-Lingual NLU._ EMNLP 2020. arXiv:2004.14353

## Synthetic data

The synthetic corpora are not in this repository either — the Skit-S2I audio
alone is roughly 11 GB. There are two ways to get them.

**Regenerate.** Every pipeline under `pipelines/` is runnable end to end and
deterministic given its seed. See [REPRODUCE.md](REPRODUCE.md). Budget roughly
1 hour for SNIPS, 11 hours for Skit-S2I, and 17 hours for MultiATIS++, dominated
by TTS.

Note that the pipelines depend on a local [Ollama](https://ollama.com) server for
text generation and translation:

```bash
ollama pull llama3.2    # generation
ollama pull llama3.3    # translation
```
