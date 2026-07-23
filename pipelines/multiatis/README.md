# MultiATIS++ Multilingual Synthesis Pipeline

A pipeline for generating synthetic **MultiATIS++ multilingual** data covering **9 languages** (EN, ES, PT, DE, FR, ZH, JA, HI, TR), including cross-lingual slot alignment, language-specific tokenization, and multilingual speech synthesis using MMS-TTS.

This pipeline extends the monolingual `multiatis_synthesis_pipeline` (English-only) to match the full MultiATIS++ dataset specification as described in the research paper.

## Pipeline Overview

```
multiatis_commands_dataset.csv (English source)
        │
        ▼
┌───────────────────────────────────┐
│ 01_translate_to_multilingual.py   │ ← Ollama LLM translation with slot value mapping
│ Input: CSV                        │
│ Output: multiatis_translated_all_languages.json
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│ 02_project_slots_crosslingual.py  │ ← BIO tagging with lang-specific tokenization
│ Input: translated JSON            │
│ Output: multiatis_bio_all_languages.json
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│ 03_synthesize_multilingual_speech.py │ ← MMS-TTS (9 language models)
│ Input: BIO-tagged JSON            │
│ Output: generated_audio/ + audio_metadata.json
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│ 04_process_multilingual_dataset.py │ ← Per-language train/dev/test splits
│ Input: BIO-tagged JSON            │
│ Output: processed_data/{lang}/{split}/
└───────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Ollama must be running for translation (Step 1)
ollama pull llama3.3

# Python dependencies
pip install ollama tqdm transformers torch soundfile scipy
```

### Step 1: Translate to all languages

```bash
python 01_translate_to_multilingual.py \
    --input data/multiatis_multilingual_pipeline/multiatis_commands_dataset.csv \
    --output data/multiatis_multilingual_pipeline/multiatis_translated_all_languages.json \
    --model llama3.3

# Quick test (5 commands only):
python 01_translate_to_multilingual.py --input ... --max-commands 5
```

### Step 2: Project slots cross-lingually

```bash
python 02_project_slots_crosslingual.py \
    --input data/multiatis_multilingual_pipeline/multiatis_translated_all_languages.json \
    --output data/multiatis_multilingual_pipeline/multiatis_bio_all_languages.json
```

### Step 3: Synthesize multilingual speech

```bash
python 03_synthesize_multilingual_speech.py \
    --jsonl data/multiatis_multilingual_pipeline/multiatis_bio_all_languages.json \
    --output_dir data/multiatis_multilingual_pipeline/generated_audio \
    --languages english spanish portuguese german french chinese japanese hindi turkish
```

### Step 4: Process into per-language train/dev/test splits

```bash
# Full data (no downsampling):
python 04_process_multilingual_dataset.py \
    --input data/multiatis_multilingual_pipeline/multiatis_bio_all_languages.json \
    --output_dir data/multiatis_multilingual_pipeline/processed_data

# With low-resource downsampling for Hindi/Turkish:
python 04_process_multilingual_dataset.py \
    --input data/multiatis_multilingual_pipeline/multiatis_bio_all_languages.json \
    --output_dir data/multiatis_multilingual_pipeline/processed_data \
    --enable_downsampling
```

### Step 5 (Optional): Check audio files

```bash
python check_audio_files.py \
    --metadata data/multiatis_multilingual_pipeline/generated_audio/audio_metadata.json
```

## Output Structure

```
processed_data/
├── train.json             # Combined train split (all languages)
├── dev.json               # Combined dev split
├── test.json              # Combined test split
├── train_metadata.json    # Audio metadata for train
├── dev_metadata.json      # Audio metadata for dev
├── test_metadata.json     # Audio metadata for test
├── intent_label.txt       # Shared intent labels
├── slot_label.txt         # Shared slot labels
├── dataset_statistics.txt # Statistics in MultiATIS++ Table 1 format
├── en/
│   ├── train/ (seq.in, seq.out, label)
│   ├── dev/
│   └── test/
├── es/
│   ├── train/ (seq.in, seq.out, label)
│   ├── dev/
│   └── test/
├── pt/ ...
├── de/ ...
├── fr/ ...
├── zh/ ...
├── ja/ ...
├── hi/ ...
└── tr/ ...
```

## Languages

| Code | Language   | Tokenization | Resource Level | Notes |
|------|-----------|--------------|----------------|-------|
| en   | English   | whitespace   | Full           | Source language |
| es   | Spanish   | whitespace   | Full           | |
| pt   | Portuguese| whitespace   | Full           | |
| de   | German    | whitespace   | Full           | |
| fr   | French    | whitespace   | Full           | |
| zh   | Chinese   | character    | Full           | Character-level BIO |
| ja   | Japanese  | character    | Full           | Character-level BIO |
| hi   | Hindi     | whitespace   | Low-resource   | Optional downsampling |
| tr   | Turkish   | whitespace   | Low-resource   | Optional downsampling |

## Key Design Decisions

### Cross-lingual Slot Alignment
Slots are aligned **semantically**, not positionally. The translation step returns entity value mappings, and the BIO projection step finds those values in the target-language text. This matches the MultiATIS++ paper's approach.

### Hindi/Turkish Low-Resource Simulation
By default, full data is generated for all languages. Use `--enable_downsampling` flag in Step 4 to downsample Hindi and Turkish to match the paper's ratios (HI: train=1440, TR: train=578). This is configurable via `config/language_config.json`.

### MMS-TTS Model Selection
Uses Facebook MMS-TTS (`facebook/mms-tts-{lang}`) for multilingual speech synthesis. Each language has its own dedicated model checkpoint loaded via Hugging Face Transformers, supporting all 9 MultiATIS++ languages with research-grade quality.

## Configuration

All language-specific settings (tokenization type, downsample sizes, MMS-TTS model codes) are centralized in `config/language_config.json`.

## Reference

- MultiATIS++ Paper: [MultiATIS++: Multi-Intent Natural Language Understanding for Air Travel](https://arxiv.org/abs/...)
- Original synthesis pipeline: `data/synthesis_pipeline/`
- Monolingual MultiATIS pipeline: `data/multiatis_synthesis_pipeline/`
