import json
import re
import argparse
import os
from ollama import chat
from tqdm import tqdm

# MultiATIS++ target languages (excluding English, which is the source)
MULTIATIS_LANGUAGES = {
    "Spanish": "es",
    "Portuguese": "pt",
    "German": "de",
    "French": "fr",
    "Chinese": "zh",
    "Japanese": "ja",
    "Hindi": "hi",
    "Turkish": "tr"
}

def load_language_config(config_path="config/language_config.json"):
    """Load language configuration from JSON file."""
    # Try relative path first, then absolute
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def translate_command(command, target_language, slots, intent, model="llama3.3"):
    """Translate an ATIS command to the target language using Ollama.
    
    Returns the translated text and a mapping of ALL slot values to their
    translated equivalents for cross-lingual slot projection.
    
    Args:
        command: English command text
        target_language: Target language name (e.g., "Spanish")
        slots: List of slot dicts with {slot_type, value, start, end}
        intent: Intent label
        model: Ollama model name
    """
    # Build slot value list for the prompt
    slot_lines = []
    for s in slots:
        slot_lines.append(f'  - {s["slot_type"]}: "{s["value"]}"')
    slot_section = '\n'.join(slot_lines) if slot_lines else '  (no slot values)'
    
    # Build expected output slot_translations format
    slot_trans_example = []
    for s in slots:
        slot_trans_example.append(
            f'{{"slot_type": "{s["slot_type"]}", "original": "{s["value"]}", "translated": "..."}}'
        )
    
    system_prompt = f"""You are an ATIS (Airline Travel Information System) voice command translator.
Translate the following command into {target_language}.

Original command: "{command}"
Intent: "{intent}"
Slot values to translate:
{slot_section}

Guidelines for translation:
1. Keep the same intent and meaning as the original command
2. Translate naturally into {target_language} — do NOT do word-by-word translation
3. Keep the translation SHORT and DIRECT — spoken-language style (short, imperative, conversational)
4. Use lowercase text
5. Minimal punctuation
6. Preserve the same entity values — translate proper nouns naturally:
   - City names: use the standard {target_language} name if one exists (e.g., "Nueva York" in Spanish)
   - Airline names: keep in English (e.g., "american airlines" stays as "american airlines")
   - Airport/airline codes: keep as-is (e.g., "jfk", "aa", "dl")
   - Numbers, dates, fare codes: keep as-is or translate naturally
7. For Chinese and Japanese: use the appropriate script (汉字 for Chinese, カタカナ/ひらがな/漢字 for Japanese)

You must return a JSON object with exactly two fields:
- "translation": the translated command string
- "slot_translations": array of objects, one per slot value, each with "slot_type", "original", and "translated" fields

Example for Spanish translation of "show flights from boston to denver on monday":
{{"translation": "muestra vuelos desde boston a denver el lunes", "slot_translations": [{{"slot_type": "fromloc.city_name", "original": "boston", "translated": "boston"}}, {{"slot_type": "toloc.city_name", "original": "denver", "translated": "denver"}}, {{"slot_type": "depart_date.day_name", "original": "monday", "translated": "lunes"}}]}}

Example for Chinese translation of "show flights from boston to denver":
{{"translation": "显示从波士顿到丹佛的航班", "slot_translations": [{{"slot_type": "fromloc.city_name", "original": "boston", "translated": "波士顿"}}, {{"slot_type": "toloc.city_name", "original": "denver", "translated": "丹佛"}}]}}

Return ONLY the JSON object, with no additional text or explanation.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Translate this ATIS command to {target_language}: {command}"}
    ]
    
    import time
    for attempt in range(3):
        try:
            response = chat(model=model, messages=messages)
            content = response.message.content
            # Remove any markdown code block markers
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Try direct JSON parse first
            try:
                result = json.loads(content)
                if result.get('translation', '').strip():
                    return result
            except json.JSONDecodeError:
                pass
            
            # Fallback: extract JSON from response using regex
            json_match = re.search(r'\{[^{}]*"translation"\s*:\s*"[^"]*"[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    if result.get('translation', '').strip():
                        return result
                except json.JSONDecodeError:
                    pass
            
            # Last resort: extract translation text directly
            trans_match = re.search(r'"translation"\s*:\s*"([^"]+)"', content)
            if trans_match:
                translation = trans_match.group(1)
                return {"translation": translation, "slot_translations": []}
            
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
    
    return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Translate MultiATIS++ English commands to all target languages')
    parser.add_argument('--input', '-i', type=str, default='data/multiatis_multilingual_pipeline/multiatis_commands_v3.json',
                        help='Input JSON file with English commands and BIO tags')
    parser.add_argument('--output', '-o', type=str, default='data/multiatis_multilingual_pipeline/multiatis_translated_all_languages.json',
                        help='Output file path')
    parser.add_argument('--languages', '-l', type=str, nargs='+', 
                        default=['Spanish', 'Portuguese', 'German', 'French', 'Chinese', 'Japanese', 'Hindi', 'Turkish'],
                        help='Target languages for translation')
    parser.add_argument('--model', '-m', type=str, default='llama3.3',
                        help='Ollama model to use')
    parser.add_argument('--max-commands', type=int, default=None,
                        help='Maximum number of commands to process (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output file, skipping already translated commands')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Only create directory if output has a directory component
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Read the English commands from the input JSON file
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            english_data = json.load(f)
        print(f"Loaded {len(english_data)} English commands from {args.input}")
        
        # Limit the number of commands if specified
        if args.max_commands and len(english_data) > args.max_commands:
            english_data = english_data[:args.max_commands]
            print(f"Limited to processing {args.max_commands} commands")
            
    except FileNotFoundError:
        print(f"Input file {args.input} not found!")
        return

    # Load existing translations if resuming
    existing_translations = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        for entry in existing_data:
            key = (entry.get('english_command', ''), entry.get('language_code', ''))
            if entry.get('translated_command', '').strip():
                existing_translations[key] = entry
        print(f"Loaded {len(existing_translations)} existing translations for resume")

    # Prepare the multilingual dataset
    multilingual_data = []
    
    # Process each command
    for row in tqdm(english_data, desc="Translating commands"):
        english_command = row['command']
        intent = row.get('intent', '')
        slots = row.get('slots', [])
        bio_tags = row.get('bio_tags', '')
        tokens = row.get('tokens', [])
        
        # First, add the original English command entry
        english_entry = {
            "english_command": english_command,
            "language": "English",
            "language_code": "en",
            "translated_command": english_command,
            "intent": intent,
            "slots": slots,
            "slot_translations": [{"slot_type": s["slot_type"], "original": s["value"], "translated": s["value"]} for s in slots],
            "bio_tags": bio_tags,
            "tokens": tokens,
            "token_count": len(tokens)
        }
        multilingual_data.append(english_entry)
        
        # Translate to each target language
        for language in args.languages:
            lang_code = MULTIATIS_LANGUAGES.get(language, language[:2].lower())
            
            # Skip if already translated (resume mode)
            key = (english_command, lang_code)
            if key in existing_translations:
                multilingual_data.append(existing_translations[key])
                continue
            
            result = translate_command(
                english_command,
                target_language=language,
                slots=slots,
                intent=intent,
                model=args.model
            )
            
            if result and 'translation' in result:
                multilingual_entry = {
                    "english_command": english_command,
                    "language": language,
                    "language_code": lang_code,
                    "translated_command": result['translation'],
                    "intent": intent,
                    "slots": slots,
                    "slot_translations": result.get('slot_translations', []),
                    "bio_tags": "",  # Will be computed in Step 02
                    "tokens": [],    # Will be computed in Step 02
                    "token_count": 0
                }
                multilingual_data.append(multilingual_entry)
            else:
                # Add entry with empty translation (will be retried later)
                multilingual_entry = {
                    "english_command": english_command,
                    "language": language,
                    "language_code": lang_code,
                    "translated_command": "",
                    "intent": intent,
                    "slots": slots,
                    "slot_translations": [],
                    "bio_tags": "",
                    "tokens": [],
                    "token_count": 0
                }
                multilingual_data.append(multilingual_entry)
                print(f"  Warning: Failed to translate to {language}, added empty entry")
    
    # Save the multilingual dataset as JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(multilingual_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nMultilingual translations generated and saved to {args.output}")
    print(f"Total entries: {len(multilingual_data)}")
    print(f"Languages: ['English'] + {args.languages}")
    print(f"English commands: {len(english_data)} (original)")
    print(f"Translation mode: 1 translation per language (matching MultiATIS++)")
    
    # Language breakdown
    language_counts = {}
    empty_counts = {}
    for entry in multilingual_data:
        lang = entry['language']
        language_counts[lang] = language_counts.get(lang, 0) + 1
        if not entry.get('translated_command', '').strip():
            empty_counts[lang] = empty_counts.get(lang, 0) + 1
    
    print("\nLanguage breakdown:")
    for lang, count in language_counts.items():
        empty = empty_counts.get(lang, 0)
        suffix = f" ({empty} empty)" if empty > 0 else ""
        print(f"  {lang}: {count} entries{suffix}")

if __name__ == "__main__":
    main()
