# Skit-S2I Synthesis Pipeline

This pipeline generates synthetic speech data similar to the **skit-s2i (Speech-to-Intent) dataset** for banking domain voice assistants.

## Dataset Overview

The skit-s2i dataset characteristics:
- **Domain**: Banking/Financial Services
- **Intents**: 14 banking intents
- **Samples**: ~11,845 total (10,445 train + 1,400 test)
- **Language**: English (Indian accent context)
- **Classification**: Intent-only (no entity slots)
- **Audio**: 8kHz original, resampled to 16kHz for training

## Intent Classes

| Class | Intent | Description |
|-------|--------|-------------|
| 0 | branch_address | Bank branch location queries |
| 1 | activate_card | Card activation requests |
| 2 | past_transactions | Transaction history inquiries |
| 3 | dispatch_status | Card/document dispatch status |
| 4 | outstanding_balance | Outstanding dues queries |
| 5 | card_issue | Card problem reports |
| 6 | ifsc_code | IFSC code queries |
| 7 | generate_pin | PIN generation requests |
| 8 | unauthorised_transaction | Fraud reports |
| 9 | loan_query | Loan-related queries |
| 10 | balance_enquiry | Account balance checks |
| 11 | change_limit | Transaction limit changes |
| 12 | block | Card blocking requests |
| 13 | lost | Lost card reports |

## Pipeline Steps

### 1. Generate Text Commands

Generate banking voice commands using Ollama LLM:

```bash
cd data/skit_s2i_synthesis_pipeline

# Generate ~11,900 samples (850 per intent × 14 intents)
python generate_monolingual_SISE_data.py \
    --num-examples 850 \
    --batch-size 50 \
    --model llama3.2 \
    --output banking_commands_dataset.csv

# Resume if interrupted
python generate_monolingual_SISE_data.py \
    --num-examples 850 \
    --batch-size 50 \
    --output banking_commands_dataset.csv \
    --resume
```

### 2. Convert to JSON Format

Convert CSV to JSON format for audio synthesis:

```bash
python generate_multilingual_SISE_data.py \
    --input banking_commands_dataset.csv \
    --output banking_commands.json \
    --pretty
```

### 3. Synthesize Speech

Generate audio files using ParlerTTS:

```bash
# Full synthesis (requires GPU for reasonable speed)
python synthesize_speech.py \
    --input banking_commands.json \
    --output-dir generated_audio \
    --speakers 11 \
    --model parler-tts/parler-tts-large-v1

# Resume from specific index
python synthesize_speech.py \
    --input banking_commands.json \
    --output-dir generated_audio \
    --start-idx 5000

# Process limited samples for testing
python synthesize_speech.py \
    --input banking_commands.json \
    --output-dir generated_audio \
    --max-samples 100
```

### 4. Validate Audio Files

Check generated audio files:

```bash
python check_audio_files.py \
    --audio-dir generated_audio \
    --output audio_check_report.json

# Fix (remove) invalid files
python check_audio_files.py \
    --audio-dir generated_audio \
    --fix
```

### 5. Create Train/Test Splits

Process the dataset into the format expected by baseline models:

```bash
python process_skit_s2i_dataset.py \
    --metadata generated_audio/audio_metadata.json \
    --audio-dir generated_audio \
    --output-dir output \
    --train-ratio 0.885
```

This creates:
- `output/train.csv` - Training set (~88.5%)
- `output/test.csv` - Test set (~11.5%)
- `output/intent_info.csv` - Intent class mappings
- `output/speaker_info.csv` - Speaker information

## Output Format

The output CSV files have the following columns:

| Column | Description |
|--------|-------------|
| id | Unique sample identifier |
| intent_class | Integer class label (0-13) |
| template | Text utterance |
| audio_path | Path to audio file |
| speaker_id | Speaker identifier |

This format is compatible with the baseline training scripts in `speech-to-intent-dataset/baselines/`.

## Requirements

### Text Generation
- Python 3.8+
- ollama
- tqdm

### Speech Synthesis
- Python 3.8+
- torch
- parler-tts
- transformers
- soundfile
- scipy
- tqdm

Install dependencies:

```bash
pip install ollama tqdm
pip install torch parler-tts transformers soundfile scipy
```

Ensure Ollama is running:

```bash
ollama serve
ollama pull llama3.2
```

## File Descriptions

| File | Description |
|------|-------------|
| `generate_monolingual_SISE_data.py` | Generate text commands using LLM |
| `generate_multilingual_SISE_data.py` | Convert CSV to JSON format |
| `synthesize_speech.py` | Generate audio using ParlerTTS |
| `check_audio_files.py` | Validate audio files |
| `process_skit_s2i_dataset.py` | Create train/test splits |

## Speaker Variations

The pipeline uses 11 speaker variations to match the original skit-s2i speaker diversity:
- 8 female speakers
- 3 male speakers
- Various simulated Indian accents

## Training Baseline Models

After generating the synthetic dataset, you can train baseline models:

```bash
cd speech-to-intent-dataset/baselines

# Train Whisper-based model
python train_whisper.py \
    --train_csv ../data/skit_s2i_synthesis_pipeline/output/train.csv \
    --test_csv ../data/skit_s2i_synthesis_pipeline/output/test.csv \
    --num_classes 14

# Train Wav2Vec2-based model
python train_wav2vec2.py \
    --train_csv ../data/skit_s2i_synthesis_pipeline/output/train.csv \
    --test_csv ../data/skit_s2i_synthesis_pipeline/output/test.csv \
    --num_classes 14
```

## Notes

1. **GPU Recommended**: Speech synthesis is much faster with a CUDA-compatible GPU
2. **Batch Processing**: Text generation uses batching to improve efficiency
3. **Resume Support**: Both text generation and speech synthesis support resuming
4. **Validation**: Always validate audio files before training to remove corrupted files
5. **Speaker Balance**: Speakers are assigned round-robin to ensure even distribution
