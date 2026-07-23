"""
Synthesize speech from text commands using ParlerTTS, then post-process each
clip to match the original Skit-S2I telephony profile (8 kHz, band-limited,
mu-law codec artifacts).

Original Skit-S2I audio profile (skit-s2i/README.md):
- 8 kHz mono 16-bit, recorded over telephone calls
- Indian English speakers (8 Female + 3 Male)
- Mean duration ~4.2 s

Pipeline per utterance:
  ParlerTTS (44.1 kHz studio quality)
      -> resample to 8 kHz
      -> telephone band-pass 300-3400 Hz (Butterworth, scipy.signal)
      -> mu-law encode + decode (G.711) for codec artifacts
      -> save as 8 kHz 16-bit mono WAV

Speaker descriptions: 11 fixed prompts indexed by speaker_id 1..11, aligned 1:1
with skit-s2i/speaker_info.csv (real distribution: 8 Female + 3 Male,
Hindi x4, Bengali x3, Kannada x2, Malayalam x1, Punjabi x1).

Each speaker description includes "speaking on a telephone call" + "natural
conversational pauses" hints to encourage longer, telephony-style delivery
and bring the mean duration closer to the original 4.2 s.

Usage:
    python synthesize_speech.py --input banking_commands.json \
        --output-dir generated_audio
"""

import argparse
import json
import os

import soundfile as sf
import torch
from tqdm import tqdm

# parler_tts / transformers are imported lazily inside load_model so this module
# stays importable in environments that only need the telephony post-process
# (for example the F5-TTS voice-cloning venv).
from slu_gap.telephony import TELEPHONE_SR, apply_telephony_postprocess

# 11 fixed speaker descriptions aligned 1:1 with the Skit-S2I speaker_info.csv.
# Indexed by speaker_id 1..11 (NOT 0-indexed).
# Distribution: 8 Female + 3 Male; Hindi x4, Bengali x3, Kannada x2, Malayalam x1, Punjabi x1.
SPEAKER_DESCRIPTIONS = {
    1: "A male speaker with a clear Indian English accent influenced by Hindi, speaking on a telephone call at a moderate pace with a friendly tone, with natural conversational pauses between phrases.",
    2: "A female speaker with a soft Indian English accent with Bengali inflection, speaking on a telephone call calmly and clearly at a measured pace, with natural pauses and a slightly distant microphone.",
    3: "A female speaker with an Indian English accent influenced by Kannada, speaking on a telephone call at a moderate pace with a polite, professional tone and natural pauses.",
    4: "A female speaker with an Indian English accent influenced by Hindi, speaking on a telephone call at a relaxed conversational pace with a warm tone and natural breath pauses.",
    5: "A female speaker with an Indian English accent with Punjabi inflection, speaking on a telephone call at a slightly fast pace with a confident, friendly tone and brief natural pauses.",
    6: "A female speaker with an Indian English accent with Bengali inflection, speaking on a telephone call at a moderate pace with a clear, pleasant tone and natural conversational pauses.",
    7: "A female speaker with an Indian English accent influenced by Malayalam, speaking on a telephone call at a measured pace with a gentle, soft tone and natural pauses.",
    8: "A male speaker with an Indian English accent influenced by Kannada, speaking on a telephone call at a moderate pace with a calm, professional tone and natural pauses between phrases.",
    9: "A female speaker with an Indian English accent influenced by Hindi, speaking on a telephone call at a moderate pace with a clear, polite tone and natural conversational pauses.",
    10: "A male speaker with an Indian English accent influenced by Hindi, speaking on a telephone call at a relaxed pace with a friendly, conversational tone and natural pauses.",
    11: "A female speaker with an Indian English accent with Bengali inflection, speaking on a telephone call at a moderate pace with a warm, professional tone and natural conversational pauses.",
}



def load_model(model_name, device="cuda"):
    """Load ParlerTTS model and tokenizer with low_cpu_mem_usage for tight-memory hosts."""
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    print(f"Loading ParlerTTS model: {model_name} ...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name, low_cpu_mem_usage=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully")
    return model, tokenizer


def synthesize_audio(model, tokenizer, text, speaker_description, device="cuda"):
    """Run ParlerTTS for one utterance. Returns (float32 audio array, src_sr).

    Wrapped in torch.no_grad() to avoid building autograd graphs across calls; we
    also explicitly delete GPU tensors so PyTorch can free them. Periodic
    torch.cuda.empty_cache() is called in the main loop (every N utterances).
    """
    with torch.no_grad():
        input_ids = tokenizer(speaker_description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.detach().cpu().to(torch.float32).numpy().squeeze()
        del input_ids, prompt_input_ids, generation
    return audio_arr, model.config.sampling_rate




def main():
    parser = argparse.ArgumentParser(description="Synthesize banking commands with telephony profile")
    parser.add_argument("--input", type=str, default="banking_commands.json",
                        help="Input JSON file with commands")
    parser.add_argument("--output-dir", type=str, default="generated_audio",
                        help="Output directory for audio files")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start index for resuming synthesis")
    parser.add_argument("--end-index", type=int, default=None,
                        help="End index for synthesis (optional)")
    parser.add_argument("--num-speakers", type=int, default=11,
                        help="Number of speakers to use (default: 11)")
    parser.add_argument("--model", type=str, default="parler-tts/parler-tts-mini-v1",
                        help="ParlerTTS model name (default: parler-tts/parler-tts-mini-v1; use parler-tts-large-v1 if C: disk has plenty of space)")
    parser.add_argument("--target-sr", type=int, default=TELEPHONE_SR,
                        help="Target sample rate for post-processed audio (default: 8000)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading commands from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} commands")

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, using CPU. This will be slow.")

    model, tokenizer = load_model(args.model, device=device)

    start_idx = args.start_index
    end_idx = args.end_index if args.end_index else len(data)
    print(f"Synthesizing audio for commands {start_idx} to {end_idx}...")
    print(f"Post-processing to {args.target_sr} Hz telephony profile (band-pass + mu-law)")

    # Fixed speaker mapping: each speaker_id 1..N is a unique voice across the whole dataset
    num_speakers = max(1, min(args.num_speakers, len(SPEAKER_DESCRIPTIONS)))
    speaker_ids = list(SPEAKER_DESCRIPTIONS.keys())[:num_speakers]

    metadata = []
    metadata_file = os.path.join(args.output_dir, "audio_metadata.json")

    if os.path.exists(metadata_file) and start_idx > 0:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} existing metadata entries")

    for idx in tqdm(range(start_idx, end_idx), desc="Synthesizing"):
        entry = data[idx]
        speaker_id = speaker_ids[idx % len(speaker_ids)]
        speaker_description = SPEAKER_DESCRIPTIONS[speaker_id]
        try:
            audio_arr, src_sr = synthesize_audio(
                model, tokenizer,
                entry["command"],
                speaker_description,
                device,
            )
            pcm_int16 = apply_telephony_postprocess(
                audio_arr, src_sr, target_sr=args.target_sr, seed=idx,
            )

            audio_filename = f"audio_{idx:06d}.wav"
            audio_path = os.path.join(args.output_dir, audio_filename)
            sf.write(audio_path, pcm_int16, args.target_sr, subtype="PCM_16")

            duration = len(pcm_int16) / float(args.target_sr)
            metadata.append({
                "id": idx,
                "file": audio_filename,
                "command": entry["command"],
                "intent": entry["intent"],
                "category": entry.get("category", "Banking"),
                "language": entry.get("language", "English"),
                "speaker_id": speaker_id,
                "sampling_rate": args.target_sr,
                "duration": duration,
            })
            if (idx + 1) % 100 == 0:
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                # Periodically free cached CUDA tensors to avoid slow OOM creep.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            # Hard fail on OOM so we know to investigate; silently skipping
            # thousands of utts is what burned us last run.
            print(f"FATAL CUDA OOM at command {idx}: {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"Error synthesizing command {idx}: {e}", flush=True)
            continue

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSynthesis complete.")
    print(f"Generated {len(metadata)} audio files in {args.output_dir}")
    print(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
