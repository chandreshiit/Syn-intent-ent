#!/usr/bin/env python3
"""
Combined script to process MultiATIS++ multilingual BIO-tagged data:
1. Split into train/dev/test sets (based on unique English commands, same split across all languages)
2. Apply language-specific downsampling for low-resource languages (Hindi, Turkish)
3. Generate per-language BIO-tagged directories
4. Generate audio metadata splits for Whisper fine-tuning
5. Output statistics matching MultiATIS++ Table 1 format

This script takes the JSON output from 02_project_slots_crosslingual.py and creates:
- {lang}/train/ {lang}/dev/ {lang}/test/ (per-language BIO-tagged format)
- train_metadata.json / dev_metadata.json / test_metadata.json (Whisper audio metadata)
"""

import json
import os
import argparse
import random
import re
from collections import defaultdict, Counter
import sys

def load_language_config(config_path="config/language_config.json"):
    """Load language configuration from JSON file."""
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def simple_word_tokenize(text):
    """
    Simple word tokenizer that splits on whitespace and punctuation.
    This replaces nltk.word_tokenize to avoid external dependencies.
    """
    tokens = re.findall(r"\w+(?:'\w+)?", text)
    return tokens

def tokenize_by_language(text, language_code, lang_config):
    """Tokenize text according to language-specific rules."""
    tokenization = "whitespace"
    
    for lang_key, lang_info in lang_config.get("languages", {}).items():
        if lang_info.get("code") == language_code:
            tokenization = lang_info.get("tokenization", "whitespace")
            break
    
    if tokenization == "character":
        tokens = [char for char in text if char.strip()]
        return tokens
    else:
        return simple_word_tokenize(text.lower())

_EN_CONTRACTIONS = {
    "'s": " is", "'ve": " have", "'t": " not", "'re": " are",
    "'m": " am", "'ll": " will", "'d": " would", "n't": " not",
}


def normalize_text(text, language_code="en"):
    """Normalize text by lowercasing. English contraction expansion only for
    English — applying it to other languages corrupts apostrophe-glued tokens
    like Turkish "detroit'den" -> "detroit woulden".

    Also collapses internal whitespace (including newlines) to single spaces.
    Some Google-translate outputs contain literal `\\n` characters which would
    otherwise produce multi-line seq.in entries and break BIO alignment.
    """
    text = text.lower()
    if language_code == "en":
        for contraction, expansion in _EN_CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
    text = re.sub(r'(\w)\.', r'\1', text)
    # Collapse any whitespace run (incl. \n, \t) to a single space.
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_audio_metadata_from_multilingual(data, audio_dir="data/multiatis_multilingual_pipeline/generated_audio"):
    """
    Create audio metadata format expected by Whisper fine-tuning from multilingual JSON data.
    """
    audio_metadata = []
    
    for i, item in enumerate(data):
        lang = item.get('language_code', item.get('language', 'unknown')).lower()
        if lang in ('english',):
            lang = 'en'
        command_id = f"cmd_{i:04d}_{lang}"
        audio_filename = f"{command_id}.wav"
        
        metadata_entry = {
            "command_id": command_id,
            "original_command": item.get('english_command', ''),
            "language": lang,
            "text": item.get('translated_command', ''),
            "intent": item.get('intent', ''),
            "entity": item.get('entity', ''),
            "category": item.get('category', ''),
            "audio_file": audio_filename,
            "audio_length": 3.0,
            "sampling_rate": 16000
        }
        
        audio_metadata.append(metadata_entry)
    
    return audio_metadata

def split_by_english_commands(data, lang_config=None, random_seed=42):
    """
    Split data by unique English commands to ensure no overlap between train/dev/test.
    Uses EXACT MultiATIS++ split sizes: train=4488, dev=490, test=893 (total=5871)
    
    CRITICAL: The same English command must go to the same split across ALL languages.
    This ensures cross-lingual evaluation integrity.
    """
    random.seed(random_seed)
    
    # Target sizes from original ATIS dataset
    TARGET_TRAIN = 4488
    TARGET_DEV = 490
    TARGET_TEST = 893
    TARGET_TOTAL = TARGET_TRAIN + TARGET_DEV + TARGET_TEST  # 5871
    
    # Group data by English command
    english_command_groups = defaultdict(list)
    for item in data:
        english_cmd = item.get('english_command', '')
        if english_cmd:
            english_command_groups[english_cmd].append(item)
    
    unique_commands = list(english_command_groups.keys())
    random.shuffle(unique_commands)
    
    # Trim to target total if we have more
    if len(unique_commands) > TARGET_TOTAL:
        print(f"  Trimming {len(unique_commands)} commands to {TARGET_TOTAL} to match ATIS")
        unique_commands = unique_commands[:TARGET_TOTAL]
    
    # Calculate split sizes based on what we have
    available = len(unique_commands)
    if available >= TARGET_TOTAL:
        test_size = TARGET_TEST
        dev_size = TARGET_DEV
    else:
        # Scale proportionally if fewer commands
        test_size = max(1, int(available * TARGET_TEST / TARGET_TOTAL))
        dev_size = max(1, int(available * TARGET_DEV / TARGET_TOTAL))
    
    test_commands = set(unique_commands[:test_size])
    dev_commands = set(unique_commands[test_size:test_size + dev_size])
    train_commands = set(unique_commands[test_size + dev_size:])
    
    # Split data based on command membership
    train_data = []
    dev_data = []
    test_data = []
    
    for cmd in train_commands:
        train_data.extend(english_command_groups[cmd])
    
    for cmd in dev_commands:
        dev_data.extend(english_command_groups[cmd])
    
    for cmd in test_commands:
        test_data.extend(english_command_groups[cmd])
    
    print(f"Split summary (targeting ATIS: {TARGET_TRAIN}/{TARGET_DEV}/{TARGET_TEST}):")
    print(f"  - Total unique English commands used: {len(unique_commands)}")
    print(f"  - Train commands: {len(train_commands)}")
    print(f"  - Dev commands: {len(dev_commands)}")
    print(f"  - Test commands: {len(test_commands)}")
    print(f"  - Train entries: {len(train_data)}")
    print(f"  - Dev entries: {len(dev_data)}")
    print(f"  - Test entries: {len(test_data)}")
    
    return train_data, dev_data, test_data, train_commands, dev_commands, test_commands

def apply_low_resource_downsampling(split_data, split_name, language_code, lang_config):
    """
    Apply downsampling for low-resource languages (Hindi, Turkish) to match
    the MultiATIS++ paper's dataset sizes.
    
    Option 3: Keep full data by default but downsample when enable_downsampling=True.
    Downsampling removes specific intents and randomly subsamples to target size.
    """
    lang_info = None
    for lang_key, info in lang_config.get("languages", {}).items():
        if info.get("code") == language_code:
            lang_info = info
            break
    
    if not lang_info or lang_info.get("full_resource", True):
        return split_data  # No downsampling needed for full-resource languages
    
    downsample_config = lang_info.get("downsample", {})
    target_size = downsample_config.get(split_name)
    excluded_intent = lang_info.get("excluded_intent")
    
    if not target_size:
        return split_data
    
    # Step 1: Remove excluded intents
    filtered_data = split_data
    if excluded_intent:
        filtered_data = [item for item in split_data 
                        if item.get('intent', '').lower() != excluded_intent.lower()]
    
    # Step 2: Downsample to target size if needed
    if len(filtered_data) > target_size:
        random.shuffle(filtered_data)
        filtered_data = filtered_data[:target_size]
    
    return filtered_data

def write_bio_files(output_dir, split_name, data, language_code, lang_config, include_language_info=False):
    """Write seq.in, label, and seq.out files for a language-specific split."""
    split_dir = os.path.join(output_dir, language_code, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # Write seq.in (tokenized commands)
    with open(os.path.join(split_dir, 'seq.in'), 'w', encoding='utf-8') as f:
        for item in data:
            command = item.get('translated_command', item.get('english_command', ''))
            if language_code in ('zh', 'ja'):
                # Character-level tokenization for CJK. `char.strip()` already
                # filters out whitespace (incl. newlines), so per-line writes
                # collapse to a single seq.in row.
                tokens = [char for char in command if char.strip()]
                f.write(' '.join(tokens) + '\n')
            else:
                f.write(normalize_text(command, language_code) + '\n')
    
    # Write label (normalized intents)
    with open(os.path.join(split_dir, 'label'), 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item['intent'].lower() + '\n')
    
    # Write seq.out (BIO tags)
    with open(os.path.join(split_dir, 'seq.out'), 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item.get('bio_tags', 'O') + '\n')
    
    if include_language_info:
        with open(os.path.join(split_dir, 'language.txt'), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(item.get('language', 'unknown') + '\n')
        
        with open(os.path.join(split_dir, 'english_command.txt'), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(item.get('english_command', 'unknown') + '\n')

def create_label_files(output_dir, processed_data):
    """Create intent_label.txt and slot_label.txt files"""
    intents = sorted(set(item['intent'].lower() for item in processed_data))
    intents = ['UNK'] + intents
    
    with open(os.path.join(output_dir, 'intent_label.txt'), 'w', encoding='utf-8') as f:
        for intent in intents:
            f.write(intent + '\n')
    
    all_bio_tags = []
    for item in processed_data:
        all_bio_tags.extend(item.get('bio_tags', 'O').split())
    unique_bio_tags = sorted(set(all_bio_tags))
    unique_bio_tags = ['PAD', 'UNK'] + unique_bio_tags
    
    with open(os.path.join(output_dir, 'slot_label.txt'), 'w', encoding='utf-8') as f:
        for tag in unique_bio_tags:
            f.write(tag + '\n')

def write_statistics(data, all_splits, output_dir, lang_config):
    """Write comprehensive dataset statistics matching MultiATIS++ Table 1 format."""
    stats_file = os.path.join(output_dir, 'dataset_statistics.txt')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("MultiATIS++ Synthetic Multilingual Dataset Statistics\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total entries: {len(data)}\n\n")
        
        # Table 1 format: Per-language statistics
        f.write("Per-Language Statistics (MultiATIS++ Table 1 format):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Language':<15} {'Train':<8} {'Dev':<8} {'Test':<8} {'Intents':<10} {'Slots':<8}\n")
        f.write("-" * 70 + "\n")
        
        # Get all language codes
        lang_codes = sorted(set(item.get('language_code', 'en') for item in data))
        
        for lang_code in lang_codes:
            train_count = sum(1 for item in all_splits.get('train', {}).get(lang_code, []))
            dev_count = sum(1 for item in all_splits.get('dev', {}).get(lang_code, []))
            test_count = sum(1 for item in all_splits.get('test', {}).get(lang_code, []))
            
            # Count unique intents and slots for this language
            lang_data = [item for item in data if item.get('language_code', '') == lang_code]
            unique_intents = len(set(item['intent'].lower() for item in lang_data))
            
            all_tags = set()
            for item in lang_data:
                for tag in item.get('bio_tags', 'O').split():
                    if tag.startswith('B-') or tag.startswith('I-'):
                        slot = tag[2:]
                        all_tags.add(slot)
            unique_slots = len(all_tags) if all_tags else 0
            
            # Look up language name
            lang_name = lang_code
            for key, info in lang_config.get("languages", {}).items():
                if info.get("code") == lang_code:
                    lang_name = info.get("name", lang_code)
                    break
            
            f.write(f"{lang_name:<15} {train_count:<8} {dev_count:<8} {test_count:<8} {unique_intents:<10} {unique_slots:<8}\n")
        
        f.write("\n")
        
        # Overall intent distribution
        intent_counts = Counter(item['intent'].lower() for item in data)
        f.write("Intent Distribution:\n")
        f.write("-" * 40 + "\n")
        for intent, count in sorted(intent_counts.items()):
            percentage = (count / len(data)) * 100
            f.write(f"{intent:<25} {count:>6} ({percentage:>5.1f}%)\n")
        f.write("\n")
        
        # Language distribution
        lang_counts = Counter(item.get('language', 'unknown') for item in data)
        f.write("Language Distribution:\n")
        f.write("-" * 40 + "\n")
        for lang, count in sorted(lang_counts.items()):
            percentage = (count / len(data)) * 100
            f.write(f"{lang:<15} {count:>6} ({percentage:>5.1f}%)\n")
        f.write("\n")
        
        # Entity analysis
        entities_with_tags = [item for item in data if 'B-' in item.get('bio_tags', '')]
        f.write(f"Entries with entities: {len(entities_with_tags)} ({len(entities_with_tags)/len(data)*100:.1f}%)\n")
        f.write(f"Entries without entities: {len(data) - len(entities_with_tags)} ({(len(data) - len(entities_with_tags))/len(data)*100:.1f}%)\n\n")
        
        # BIO tag distribution
        all_bio_tags = []
        for item in data:
            if 'bio_tags' in item:
                all_bio_tags.extend(item['bio_tags'].split())
        
        if all_bio_tags:
            bio_tag_counts = Counter(all_bio_tags)
            f.write("BIO Tag Distribution:\n")
            f.write("-" * 40 + "\n")
            for tag, count in sorted(bio_tag_counts.items()):
                percentage = (count / len(all_bio_tags)) * 100
                f.write(f"{tag:<35} {count:>6} ({percentage:>5.1f}%)\n")
            f.write("\n")
        
        # Text length statistics per language
        f.write("Text Length Statistics (in tokens) per Language:\n")
        f.write("-" * 50 + "\n")
        for lang_code in lang_codes:
            lang_data = [item for item in data if item.get('language_code', '') == lang_code]
            text_lengths = [item.get('token_count', len(simple_word_tokenize(item.get('translated_command', '')))) for item in lang_data]
            
            if text_lengths:
                lang_name = lang_code
                for key, info in lang_config.get("languages", {}).items():
                    if info.get("code") == lang_code:
                        lang_name = info.get("name", lang_code)
                        break
                
                f.write(f"{lang_name}: avg={sum(text_lengths)/len(text_lengths):.1f}, "
                       f"min={min(text_lengths)}, max={max(text_lengths)}\n")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process MultiATIS++ multilingual BIO-tagged data into per-language train/dev/test splits')
    parser.add_argument('--input', '-i', type=str, default='data/multiatis_multilingual_pipeline/multiatis_bio_all_languages.json',
                        help='Input BIO-tagged JSON file')
    parser.add_argument('--output_dir', '-o', type=str, default='data/multiatis_multilingual_pipeline/processed_data',
                        help='Output directory for all generated files')
    parser.add_argument('--audio_dir', type=str, default='data/multiatis_multilingual_pipeline/generated_audio',
                        help='Audio directory path for metadata generation')
    parser.add_argument('--config', '-c', type=str, default='config/language_config.json',
                        help='Language configuration file')
    parser.add_argument('--test_ratio', '-t', type=float, default=0.152,
                        help='Ratio of data for testing (default: 0.152 for MultiATIS++)')
    parser.add_argument('--dev_ratio', '-d', type=float, default=0.083,
                        help='Ratio of data for dev set (default: 0.083 for MultiATIS++)')
    parser.add_argument('--enable_downsampling', action='store_true',
                        help='Enable downsampling for low-resource languages (Hindi, Turkish)')
    parser.add_argument('--include_language_info', action='store_true',
                        help='Include language and english_command information in BIO output files')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducible splits')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Load language config
    lang_config = load_language_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input data
    print(f"Loading data from {args.input}...")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)
    
    # Ensure BIO tags exist for all entries
    for item in data:
        if 'bio_tags' not in item:
            command = item.get('translated_command', item.get('english_command', ''))
            tokens = simple_word_tokenize(normalize_text(command))
            item['bio_tags'] = ' '.join(['O'] * len(tokens))
    
    # Split data (same split across ALL languages, based on English command).
    # The splitter uses MultiATIS++ exact targets (4488/490/893) internally; the
    # CLI test_ratio/dev_ratio kwargs from older versions are ignored.
    print(f"\nSplitting data using MultiATIS++ exact targets (4488/490/893)...")
    train_data, dev_data, test_data, train_cmds, dev_cmds, test_cmds = split_by_english_commands(
        data,
        random_seed=args.random_seed,
    )
    
    # Get all language codes in the data
    lang_codes = sorted(set(item.get('language_code', 'en') for item in data))
    
    # Resolve language codes for entries that use language names
    lang_name_to_code = {
        'english': 'en', 'spanish': 'es', 'portuguese': 'pt',
        'german': 'de', 'french': 'fr', 'chinese': 'zh',
        'japanese': 'ja', 'hindi': 'hi', 'turkish': 'tr'
    }
    for item in data:
        if 'language_code' not in item or not item['language_code']:
            lang_name = item.get('language', '').lower()
            item['language_code'] = lang_name_to_code.get(lang_name, 'en')
    
    # Rebuild lang_codes after resolution
    lang_codes = sorted(set(item.get('language_code', 'en') for item in data))
    
    print(f"\nLanguages found: {lang_codes}")
    
    # Process per-language splits
    all_splits = {'train': {}, 'dev': {}, 'test': {}}
    
    for lang_code in lang_codes:
        print(f"\nProcessing {lang_code}...")
        
        # Filter split data by language
        lang_train = [item for item in train_data if item.get('language_code', 'en') == lang_code]
        lang_dev = [item for item in dev_data if item.get('language_code', 'en') == lang_code]
        lang_test = [item for item in test_data if item.get('language_code', 'en') == lang_code]
        
        # Apply downsampling for low-resource languages if enabled
        if args.enable_downsampling:
            lang_train = apply_low_resource_downsampling(lang_train, 'train', lang_code, lang_config)
            lang_dev = apply_low_resource_downsampling(lang_dev, 'dev', lang_code, lang_config)
            lang_test = apply_low_resource_downsampling(lang_test, 'test', lang_code, lang_config)
        
        all_splits['train'][lang_code] = lang_train
        all_splits['dev'][lang_code] = lang_dev
        all_splits['test'][lang_code] = lang_test
        
        print(f"  {lang_code}: train={len(lang_train)}, dev={len(lang_dev)}, test={len(lang_test)}")
        
        # Write per-language BIO format files
        write_bio_files(args.output_dir, 'train', lang_train, lang_code, lang_config, args.include_language_info)
        write_bio_files(args.output_dir, 'dev', lang_dev, lang_code, lang_config, args.include_language_info)
        write_bio_files(args.output_dir, 'test', lang_test, lang_code, lang_config, args.include_language_info)
    
    # Write JSON splits (all languages combined)
    print("\nWriting JSON splits...")
    with open(os.path.join(args.output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, 'dev.json'), 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # Create label files (shared across all languages)
    print("Creating label files...")
    create_label_files(args.output_dir, data)
    
    # Generate audio metadata splits
    print("Generating audio metadata splits...")
    audio_metadata = create_audio_metadata_from_multilingual(data, args.audio_dir)
    
    train_audio = [m for m in audio_metadata if m.get('original_command', '') in train_cmds]
    dev_audio = [m for m in audio_metadata if m.get('original_command', '') in dev_cmds]
    test_audio = [m for m in audio_metadata if m.get('original_command', '') in test_cmds]
    
    with open(os.path.join(args.output_dir, 'train_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(train_audio, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, 'dev_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(dev_audio, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, 'test_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(test_audio, f, ensure_ascii=False, indent=2)
    
    # Write statistics
    print("Writing dataset statistics...")
    write_statistics(data, all_splits, args.output_dir, lang_config)
    
    downsampling_note = " (downsampling ENABLED)" if args.enable_downsampling else " (downsampling DISABLED - full data for all languages)"
    
    print(f"\nProcessing complete!{downsampling_note}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"  +-- train.json")
    print(f"  +-- dev.json")
    print(f"  +-- test.json")
    print(f"  +-- train_metadata.json")
    print(f"  +-- dev_metadata.json")
    print(f"  +-- test_metadata.json")
    print(f"  +-- intent_label.txt")
    print(f"  +-- slot_label.txt")
    print(f"  +-- dataset_statistics.txt")
    for lang_code in lang_codes:
        print(f"  +-- {lang_code}/")
        print(f"  |   +-- train/ (seq.in, seq.out, label)")
        print(f"  |   +-- dev/ (seq.in, seq.out, label)")
        print(f"  |   +-- test/ (seq.in, seq.out, label)")

if __name__ == "__main__":
    main()
