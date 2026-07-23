#!/usr/bin/env python3
"""
Cross-lingual slot projection for MultiATIS++ multilingual dataset.
Takes translated utterances with slot_translations mappings and produces BIO tags
using language-specific tokenization (whitespace for most, character-level for ZH/JA).

This version supports MULTIPLE slots per utterance (full 79-type ATIS slot taxonomy).
"""

import json
import os
import argparse
import re
from collections import Counter
from tqdm import tqdm

def load_language_config(config_path="config/language_config.json"):
    """Load language configuration from JSON file."""
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def tokenize_by_language(text, language_code, lang_config):
    """Tokenize text according to language-specific rules.
    
    - Whitespace tokenization for: EN, ES, PT, DE, FR, HI, TR
    - Character-level tokenization for: ZH, JA
    """
    tokenization = "whitespace"
    
    # Look up tokenization type from config
    for lang_key, lang_info in lang_config.get("languages", {}).items():
        if lang_info.get("code") == language_code or lang_info.get("name", "").lower() == language_code.lower():
            tokenization = lang_info.get("tokenization", "whitespace")
            break
    
    if tokenization == "character":
        # Character-level tokenization for Chinese and Japanese
        # Each character is a separate token (no spaces between characters)
        tokens = []
        for char in text:
            if char.strip():  # Skip whitespace characters
                tokens.append(char)
        return tokens
    else:
        # Whitespace tokenization for all other languages
        tokens = re.findall(r"\w+(?:'\w+)?", text.lower())
        return tokens

def find_value_in_tokens(tokens, value, language_code, is_character_level=False):
    """Find the position of a slot value within the token list.
    
    Returns (start_idx, end_idx) tuple or None if not found.
    Tries multiple matching strategies for robustness.
    """
    if not value or value.lower() == 'none':
        return None
    
    if is_character_level:
        # Character-level matching for CJK languages
        value_chars = [c for c in value if c.strip()]
        if not value_chars:
            return None
        
        # Strategy 1: Exact character sequence match
        for i in range(len(tokens) - len(value_chars) + 1):
            if tokens[i:i + len(value_chars)] == value_chars:
                return (i, i + len(value_chars))
        
        # Strategy 2: Case-insensitive match
        lower_tokens = [t.lower() for t in tokens]
        lower_chars = [c.lower() for c in value_chars]
        for i in range(len(lower_tokens) - len(lower_chars) + 1):
            if lower_tokens[i:i + len(lower_chars)] == lower_chars:
                return (i, i + len(lower_chars))
        
        return None
    else:
        # Word-level matching for whitespace-tokenized languages
        value_tokens = re.findall(r"\w+(?:'\w+)?", value.lower())
        if not value_tokens:
            return None
        
        lower_tokens = [t.lower() for t in tokens]
        
        # Strategy 1: Exact sequence match
        for i in range(len(lower_tokens) - len(value_tokens) + 1):
            if lower_tokens[i:i + len(value_tokens)] == value_tokens:
                return (i, i + len(value_tokens))
        
        # Strategy 2: Single word partial match
        if len(value_tokens) == 1:
            for i, token in enumerate(lower_tokens):
                if token == value_tokens[0]:
                    return (i, i + 1)
        
        # Strategy 3: Substring containment for compounds
        for i, token in enumerate(lower_tokens):
            if value_tokens[0] in token:
                return (i, i + 1)
        
        return None

def create_bio_tags_multislot(item, lang_config):
    """Create BIO tags for a translated command with MULTIPLE slot projections.
    
    Uses the slot_translations field to find ALL slot positions in the target language,
    and applies language-specific tokenization.
    """
    language_code = item.get('language_code', 'en')
    command = item.get('translated_command', item.get('english_command', ''))
    slot_translations = item.get('slot_translations', [])
    
    if not command or not command.strip():
        return 'O', []
    
    # Determine tokenization type
    is_character_level = language_code in ('zh', 'ja')
    
    # Tokenize the command
    tokens = tokenize_by_language(command, language_code, lang_config)
    
    if not tokens:
        return 'O', tokens
    
    # Initialize all tags as O
    bio_tags = ['O'] * len(tokens)
    tagged_positions = set()
    
    # Build a mapping from original slot values to their translated values
    # Use the ORIGINAL slots for slot_type (always correct), and slot_translations
    # only for the translated value (LLM often garbles slot_type names)
    original_slots = item.get('slots', [])
    
    # Create lookup: original_value -> translated_value from slot_translations
    trans_lookup = {}
    for st in slot_translations:
        orig_val = (st.get('original') or '').lower().strip()
        trans_val = (st.get('translated') or '').strip()
        if orig_val and trans_val:
            trans_lookup[orig_val] = trans_val
    
    # Process each ORIGINAL slot (guaranteed correct slot_type)
    for slot in original_slots:
        slot_type = slot.get('slot_type', '')
        original_value = slot.get('value', '')
        
        if not slot_type or not original_value:
            continue
        
        # Get the translated value from lookup, fall back to original
        translated_value = trans_lookup.get(original_value.lower().strip(), original_value)
        
        # Find the translated value in tokens
        match = find_value_in_tokens(tokens, translated_value, language_code, is_character_level)
        
        # If translated value not found, try the original value
        if match is None and translated_value != original_value:
            match = find_value_in_tokens(tokens, original_value, language_code, is_character_level)
        
        if match is not None:
            start_idx, end_idx = match
            # Check for overlap with already tagged positions
            positions = set(range(start_idx, end_idx))
            if positions & tagged_positions:
                continue  # Skip overlapping slots
            
            # Apply BIO tags
            bio_tags[start_idx] = f'B-{slot_type}'
            for j in range(start_idx + 1, end_idx):
                bio_tags[j] = f'I-{slot_type}'
            tagged_positions.update(positions)
    
    return ' '.join(bio_tags), tokens

def parse_arguments():
    parser = argparse.ArgumentParser(description='Project slots cross-lingually and generate BIO tags for all languages')
    parser.add_argument('--input', '-i', type=str, default='data/multiatis_multilingual_pipeline/multiatis_translated_all_languages.json',
                        help='Input translated JSON file')
    parser.add_argument('--output', '-o', type=str, default='data/multiatis_multilingual_pipeline/multiatis_bio_all_languages.json',
                        help='Output file with BIO tags')
    parser.add_argument('--config', '-c', type=str, default='config/language_config.json',
                        help='Language configuration file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Only create directory if output has a directory component
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load language config
    lang_config = load_language_config(args.config)
    
    # Load translated data
    print(f"Loading translated data from {args.input}...")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found!")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return
    
    # Generate BIO tags for each entry
    print("Generating cross-lingual BIO tags (multi-slot projection)...")
    alignment_stats = Counter()
    slot_type_counts = Counter()
    
    for item in tqdm(data, desc="Projecting slots"):
        bio_tags, tokens = create_bio_tags_multislot(item, lang_config)
        item['bio_tags'] = bio_tags
        item['tokens'] = tokens
        item['token_count'] = len(tokens)
        
        # Track alignment statistics
        lang = item.get('language', 'unknown')
        total_slots = len(item.get('slot_translations', []))
        found_slots = sum(1 for t in bio_tags.split() if t.startswith('B-'))
        
        alignment_stats[f"{lang}_total"] += 1
        alignment_stats[f"{lang}_slots_expected"] += total_slots
        alignment_stats[f"{lang}_slots_found"] += found_slots
        
        if found_slots > 0:
            alignment_stats[f"{lang}_has_slots"] += 1
        
        # Count slot types
        for t in bio_tags.split():
            if t.startswith('B-'):
                slot_type_counts[t[2:]] += 1
        
        # Validate token count matches tag count
        tag_count = len(bio_tags.split())
        if tag_count != len(tokens):
            # Fix by padding or truncating tags
            if tag_count < len(tokens):
                item['bio_tags'] = bio_tags + ' ' + ' '.join(['O'] * (len(tokens) - tag_count))
            else:
                item['bio_tags'] = ' '.join(bio_tags.split()[:len(tokens)])
    
    # Save output
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nBIO-tagged data saved to {args.output}")
    print(f"Total entries: {len(data)}")
    print(f"Unique slot types used: {len(slot_type_counts)}")
    
    # Print alignment statistics
    print("\nSlot alignment statistics per language:")
    languages_seen = sorted(set(item.get('language', 'unknown') for item in data))
    for lang in languages_seen:
        total = alignment_stats.get(f"{lang}_total", 0)
        has_slots = alignment_stats.get(f"{lang}_has_slots", 0)
        expected = alignment_stats.get(f"{lang}_slots_expected", 0)
        found = alignment_stats.get(f"{lang}_slots_found", 0)
        if total > 0:
            pct = found / expected * 100 if expected > 0 else 0
            print(f"  {lang}: {has_slots}/{total} entries with slots, "
                  f"{found}/{expected} individual slots found ({pct:.1f}%)")

if __name__ == "__main__":
    main()
