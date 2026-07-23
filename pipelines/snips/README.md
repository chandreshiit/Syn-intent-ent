# SNIPS Smart-Lights Multilingual Synthesis Pipeline

Synthetic multilingual SNIPS smart-lights dataset (English + French) with
**multi-slot BIO annotation** preserved from source through cross-lingual projection,
and **MMS-TTS** audio at 16 kHz. Designed to match the original SNIPS smart-lights
sub-dataset exactly in intent distribution and slot structure.

This pipeline is a parallel of `data/multiatis_multilingual_pipeline/` adapted to
the SNIPS smart-lights ontology, and is **independent** of the existing
single-entity English `data/snips_synthesis_pipeline/`.

## Target statistics (matching `SNIPS/smart-lights-en-close-field/dataset.json`)

| Intent | Slot(s) | EN count | FR count |
|---|---|---|---|
| DecreaseBrightness | room | 296 | 296 |
| IncreaseBrightness | room | 296 | 296 |
| SetLightBrightness | room, brightness | 296 | 296 |
| SetLightColor | room, color | 300 | 300 |
| SwitchLightOff | room | 299 | 299 |
| SwitchLightOn | room | 278 | 278 |
| **Total** | | **1,765** | **1,765** |

Slot ontology: `room` (34 values), `color` (3 values + synonyms), `brightness`
(built-in numeric, word and digit forms ~50/50). Two intents (SetLightBrightness,
SetLightColor) carry **two slots per utterance**.

## Pipeline overview

```
config/snips_slot_taxonomy.json
config/language_config.json
        |
        v
00_generate_source_commands.py
        |  (template-based EN generation with multi-slot BIO at the source)
        v
snips_commands_v1.json
        |
        v
01_translate_to_multilingual.py
        |  (Ollama LLM EN -> FR; returns translation + slot_translations)
        v
snips_translated_all_languages.json
        |
        v
02_project_slots_crosslingual.py
        |  (BIO projection for FR via slot_translations; EN tags kept from step 00)
        v
snips_bio_all_languages.json
        |
        +--> 03_synthesize_multilingual_speech.py  (MMS-TTS eng + fra, 16 kHz)
        |          v
        |    generated_audio/  + audio_metadata.json
        |
        +--> 04_process_multilingual_dataset.py  (BIO directories + label files + stats)
                   v
             processed_data/{en,fr,combined}/all/{seq.in,seq.out,label} + label files
```

## Quick start

```bash
# 1. Generate English source commands with multi-slot BIO at the source
python 00_generate_source_commands.py \
    --config config/snips_slot_taxonomy.json \
    --output snips_commands_v1.json \
    --seed 42

# 2. Translate to target languages (default: French only)
python 01_translate_to_multilingual.py \
    --input snips_commands_v1.json \
    --output snips_translated_all_languages.json \
    --languages French \
    --model llama3.2

# 3. Cross-lingual BIO projection
python 02_project_slots_crosslingual.py \
    --input snips_translated_all_languages.json \
    --output snips_bio_all_languages.json \
    --config config/language_config.json

# 4. Synthesize speech (MMS-TTS, 16 kHz)
python 03_synthesize_multilingual_speech.py \
    --jsonl snips_bio_all_languages.json \
    --output_dir generated_audio \
    --languages english french

# 5. Process into BIO format + statistics
python 04_process_multilingual_dataset.py \
    --input snips_bio_all_languages.json \
    --output_dir processed_data \
    --audio_dir generated_audio
```

## Design choices

### Source-stage BIO is exact, not LLM-projected

Step 00 produces utterances by template substitution, so the slot spans are known
at insertion time and the BIO tags are correct by construction (0% annotation error).
This is the right call because:

- SNIPS smart-lights has only 3 slot types and a strictly bounded surface form.
- The Snips Voice paper (sec. 3.2.1) describes the same approach in its own
  data-generation pipeline.
- For French, the LLM only has to produce a fluent translation + return the
  translated slot values; step 02 then locates those values in the translated
  text to project BIO tags. This is the same pattern as
  `data/multiatis_multilingual_pipeline/02_project_slots_crosslingual.py`.

### Multi-slot support

`SetLightBrightness` (room + brightness) and `SetLightColor` (room + color) carry
two slots per utterance. The cross-lingual projection in step 02 handles multiple
slots per utterance natively and refuses overlapping spans.

### TTS: MMS-TTS, single fixed voice per language

`facebook/mms-tts-eng` and `facebook/mms-tts-fra`, 16 kHz mono. MMS-TTS is
single-speaker per checkpoint; we do not simulate the 69-speaker variation of the
original SNIPS recordings. The multilingual experiment is not aimed at speaker
generalisation; it tests cross-lingual SLU on synthetic input.

### No pre-baked train/dev/test split

The reviewer-requested 5-fold cross-validation (review R7) requires the full
dataset as input; the kfold script handles slicing. Step 04 therefore emits the
full BIO data as `{lang}/all/` and `combined/all/`, plus label and metadata files
that the existing baseline scripts can consume.

## File map

```
data/snips_multilingual_pipeline/
|-- 00_generate_source_commands.py
|-- 01_translate_to_multilingual.py
|-- 02_project_slots_crosslingual.py
|-- 03_synthesize_multilingual_speech.py
|-- 04_process_multilingual_dataset.py
|-- README.md
|-- config/
|   |-- snips_slot_taxonomy.json   # room, color, brightness ontology + intent counts
|   `-- language_config.json       # en, fr (mms_tts_code, tokenization)
|-- snips_commands_v1.json            # step 00 output
|-- snips_translated_all_languages.json  # step 01 output
|-- snips_bio_all_languages.json      # step 02 output
|-- generated_audio/                  # step 03 output (1,765 EN + 1,765 FR wavs)
`-- processed_data/                   # step 04 output (BIO + labels + stats)
```

## Reference

- Original SNIPS smart-lights: `SNIPS/smart-lights-en-close-field/dataset.json`
- Parallel multilingual pipeline (template structure / cross-lingual projection):
  `data/multiatis_multilingual_pipeline/`
- Original English-only synthetic SNIPS (single-entity, not used here):
  `data/snips_synthesis_pipeline/`
