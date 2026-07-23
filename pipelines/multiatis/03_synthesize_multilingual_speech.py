import torch
from transformers import VitsModel, AutoTokenizer, set_seed
import soundfile as sf
import scipy
import scipy.io.wavfile
import json
import os
import argparse
import io
import numpy as np
from tqdm import tqdm
from gtts import gTTS

# Set device, dtype, and other configurations
torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32  # MMS-TTS uses float32

# MMS-TTS model name mapping for MultiATIS++ languages
# Uses Facebook's Massively Multilingual Speech TTS models
# Note: zh (cmn) and ja (jpn) are gated on HuggingFace, so we use gTTS fallback
MMS_TTS_MODELS = {
    "en": "facebook/mms-tts-eng",
    "es": "facebook/mms-tts-spa",
    "pt": "facebook/mms-tts-por",
    "de": "facebook/mms-tts-deu",
    "fr": "facebook/mms-tts-fra",
    "hi": "facebook/mms-tts-hin",
    "tr": "facebook/mms-tts-tur"
}

# gTTS language codes for languages not covered by MMS-TTS
GTTS_LANGUAGES = {
    "zh": "zh-CN",
    "ja": "ja",
}

# Language name to code mapping
LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "spanish": "es",
    "portuguese": "pt",
    "german": "de",
    "french": "fr",
    "chinese": "zh",
    "japanese": "ja",
    "hindi": "hi",
    "turkish": "tr"
}

def load_jsonl(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_mms_model(language_code, token=None):
    """Load the MMS-TTS model and tokenizer for a specific language."""
    model_name = MMS_TTS_MODELS.get(language_code)
    if not model_name:
        print(f"Warning: No MMS-TTS model found for language code '{language_code}'")
        return None, None
    
    print(f"Loading MMS-TTS model for {language_code}: {model_name}")
    try:
        model = VitsModel.from_pretrained(model_name, token=token).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    except (OSError, Exception) as e:
        print(f"\n*** Failed to load model for {language_code} ({model_name}): {e}")
        print(f"*** This model may be gated. Try: huggingface-cli login")
        print(f"*** Or pass --token YOUR_HF_TOKEN to authenticate.")
        print(f"*** Skipping {language_code}...\n")
        return None, None
    
    return model, tokenizer

def generate_speech(model, tokenizer, text, output_file):
    """Generate speech for a single text input using MMS-TTS."""
    inputs = tokenizer(text, return_tensors="pt")
    # Ensure input_ids are Long type (some non-Latin scripts can cause float tensors)
    inputs = {k: v.to(torch_device).long() if v.dtype == torch.float32 and k == 'input_ids' 
              else v.to(torch_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model(**inputs).waveform
    
    # Convert to numpy
    audio_np = output.cpu().to(torch.float32).numpy().squeeze()
    
    # Get sampling rate from model config
    sampling_rate = model.config.sampling_rate
    
    # Save audio file
    try:
        sf.write(output_file, audio_np, sampling_rate)
    except Exception as e:
        print(f"Failed to write audio with soundfile: {e}")
        try:
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(-1, 1)
            scipy.io.wavfile.write(output_file, rate=sampling_rate, data=audio_np)
        except Exception as e:
            print(f"Failed to write audio with both methods: {e}")
            with open(output_file, 'wb') as f:
                pass
            print(f"Created empty audio file as fallback: {output_file}")
    
    audio_length = float(len(audio_np)) / sampling_rate
    
    return {
        "audio_file": output_file,
        "text": text,
        "sampling_rate": sampling_rate,
        "audio_length": audio_length
    }

def generate_speech_gtts(text, output_file, lang_code, max_retries=5):
    """Generate speech using Google TTS (gTTS) for languages not covered by MMS-TTS.
    
    gTTS produces mp3, so we save directly as mp3 (renamed to .wav for pipeline consistency).
    Includes retry logic with exponential backoff for 429 rate limiting.
    """
    import time as _time
    
    gtts_lang = GTTS_LANGUAGES.get(lang_code, lang_code)
    
    # Retry loop with exponential backoff for rate limiting
    for attempt in range(max_retries):
        try:
            tts = gTTS(text=text, lang=gtts_lang)
            mp3_buffer = io.BytesIO()
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)
            break  # Success
        except Exception as e:
            if '429' in str(e) and attempt < max_retries - 1:
                wait_time = min(2 ** (attempt + 1), 60)  # 2, 4, 8, 16, 60 seconds
                _time.sleep(wait_time)
                continue
            raise  # Re-raise on final attempt or non-429 error
    
    # Save as mp3 first, then try to convert to wav
    mp3_file = output_file.replace('.wav', '.mp3')
    with open(mp3_file, 'wb') as f:
        f.write(mp3_buffer.read())
    
    # Try to convert mp3 to wav using soundfile/audioread
    sampling_rate = 22050  # gTTS default
    audio_length = 0.0
    try:
        import audioread
        with audioread.audio_open(mp3_file) as f_audio:
            sampling_rate = f_audio.samplerate
            audio_length = f_audio.duration
            audio_data = b''.join(f_audio)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            sf.write(output_file, audio_np, sampling_rate)
        os.remove(mp3_file)
    except Exception:
        # If conversion fails, keep the mp3 and rename to .wav
        os.replace(mp3_file, output_file)
        audio_length = os.path.getsize(output_file) / (16000 / 8)
    
    # Small delay between requests to avoid rate limiting
    _time.sleep(0.35)
    
    return {
        "audio_file": output_file,
        "text": text,
        "sampling_rate": sampling_rate,
        "audio_length": audio_length
    }

def main(jsonl_path, languages, output_dir, resume=False, token=None):
    """Process JSON file, generate speech for each translation, and create metadata."""
    print(f"Starting MMS-TTS multilingual generation...")
    print(f"Device: {torch_device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from JSON file
    data = load_jsonl(jsonl_path)
    print(f"Loaded {len(data)} entries from {jsonl_path}")
    
    # Convert language names to codes
    lang_codes = []
    for lang in languages:
        code = LANGUAGE_NAME_TO_CODE.get(lang.lower(), lang.lower())
        lang_codes.append(code)
    
    print(f"Languages requested: {languages}")
    print(f"Language codes: {lang_codes}")
    
    # Filter data by requested languages
    filtered_data = []
    for entry in data:
        entry_lang = entry.get("language", "").lower()
        entry_code = entry.get("language_code", "")
        
        # Match by either language name or code
        lang_code = LANGUAGE_NAME_TO_CODE.get(entry_lang, entry_code)
        if lang_code in lang_codes:
            entry['_resolved_lang_code'] = lang_code
            filtered_data.append(entry)
    
    print(f"Processing {len(filtered_data)} entries from {len(data)} total entries...")
    
    # Group entries by language for efficient model loading
    entries_by_lang = {}
    for entry in filtered_data:
        lang_code = entry['_resolved_lang_code']
        if lang_code not in entries_by_lang:
            entries_by_lang[lang_code] = []
        entries_by_lang[lang_code].append(entry)
    
    # Process each language group
    metadata = []
    
    for lang_code, entries in entries_by_lang.items():
        print(f"\n{'='*60}")
        print(f"Processing {lang_code} ({len(entries)} entries)")
        print(f"{'='*60}")
        
        # Determine TTS backend for this language
        use_gtts = lang_code in GTTS_LANGUAGES
        model, tokenizer = None, None
        
        if use_gtts:
            print(f"Using gTTS (Google TTS) for {lang_code}")
        else:
            # Load MMS-TTS model for this language
            model, tokenizer = load_mms_model(lang_code, token=token)
            if model is None:
                print(f"Skipping {lang_code} - no model available")
                continue
        
        # Process entries for this language
        skipped = 0
        errors = 0
        progress_bar = tqdm(total=len(entries), desc=f"Generating {lang_code} speech", unit="audio")
        for i, entry in enumerate(entries):
            command_id = f"cmd_{i:04d}_{lang_code}"
            
            translated_text = entry.get("translated_command", "")
            english_cmd = entry.get("english_command", f"cmd_{i}")
            
            if not translated_text:
                progress_bar.update(1)
                continue
            
            output_file = os.path.join(output_dir, f"{command_id}.wav")
            
            # Resume: skip if file already exists and has content
            if resume and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                skipped += 1
                progress_bar.update(1)
                continue
            
            try:
                if use_gtts:
                    info = generate_speech_gtts(translated_text, output_file, lang_code)
                else:
                    info = generate_speech(model, tokenizer, translated_text, output_file)
                
                metadata.append({
                    "command_id": command_id,
                    "original_command": english_cmd,
                    "language": lang_code,
                    "text": translated_text,
                    "intent": entry.get("intent", ""),
                    "entity": entry.get("entity", ""),
                    "category": entry.get("category", ""),
                    "audio_file": os.path.basename(output_file),
                    "audio_length": info["audio_length"],
                    "sampling_rate": info["sampling_rate"],
                    "tts_engine": "gtts" if use_gtts else "mms-tts"
                })
            except Exception as e:
                errors += 1
                if errors <= 5:  # Only print first 5 errors per language
                    print(f"Error generating speech for entry {i} ({lang_code}): {e}")
                elif errors == 6:
                    print(f"  ... suppressing further errors for {lang_code}")
            
            progress_bar.update(1)
        
        progress_bar.close()
        if resume and skipped > 0:
            print(f"  Resumed: skipped {skipped} already-generated files for {lang_code}")
        if errors > 0:
            print(f"  Errors: {errors} entries failed for {lang_code}")
        
        # Free model memory before loading next language
        if model is not None:
            del model
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save metadata to JSON file
    metadata_file = os.path.join(output_dir, "audio_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\nGenerated {len(metadata)} audio files. Metadata saved to {metadata_file}")
    
    # Language breakdown
    lang_counts = {}
    for entry in metadata:
        lang = entry['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    print("\nLanguage breakdown:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count} audio files")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate multilingual TTS using Facebook MMS-TTS for MultiATIS++ translations')
    parser.add_argument('--jsonl', type=str, default="data/multiatis_multilingual_pipeline/multiatis_bio_all_languages.json", 
                        help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str, default="data/multiatis_multilingual_pipeline/generated_audio", 
                        help='Directory to save generated audio files')
    parser.add_argument('--languages', type=str, nargs='+', 
                        default=["english", "spanish", "portuguese", "german", "french", "chinese", "japanese", "hindi", "turkish"],
                        help='Languages to generate speech for')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for generation')
    parser.add_argument('--resume', action='store_true',
                        help='Skip generating audio files that already exist (for resuming interrupted runs)')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace access token for gated models (e.g. mms-tts-cmn, mms-tts-jpn)')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    main(args.jsonl, args.languages, args.output_dir, resume=args.resume, token=args.token)
