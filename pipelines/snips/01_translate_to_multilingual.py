#!/usr/bin/env python3
"""
Translate SNIPS smart-lights English source commands into target languages.

Mirrors data/multiatis_multilingual_pipeline/01_translate_to_multilingual.py in
prompt structure and JSON I/O shape. The only specialisations are:
  - domain wording (smart-home / smart-lights instead of ATIS / airline travel)
  - slot value translation rules (room names -> natural translation, color -> translated,
    brightness numbers -> translated word or kept as digits)
  - default target language list is just French (per project scope: EN + FR)

Output schema matches the MultiATIS pipeline so step 02 / 03 can be reused with
minimal change. Each output entry has fields:
  english_command, language, language_code, translated_command, intent,
  slots (original spans), slot_translations (target-language values for projection),
  bio_tags (filled by step 02), tokens (filled by step 02), token_count.
"""

import json
import re
import time
import argparse
import os
from ollama import chat
from tqdm import tqdm


SNIPS_LANGUAGES = {
    "French": "fr",
}


# Per-intent verb guidance per target language. The small Ollama model
# (llama3.2) regularly inverts semantics (e.g., SwitchLightOn -> "eteins" = turn off
# in French). To prevent this we list the EXPECTED FR verbs per intent and
# validate the LLM output contains at least one of them; we also flag forbidden
# (semantically-opposite) verbs and retry if found.
INTENT_VERB_GUIDANCE = {
    "fr": {
        "SwitchLightOn":      {"expected": ["allume", "allumez", "active",  "mets en marche"],
                                "forbidden": ["eteins", "eteignez", "ferme", "fermez", "desactive"]},
        "SwitchLightOff":     {"expected": ["eteins", "eteignez", "ferme"],
                                "forbidden": ["allume", "allumez"]},
        "IncreaseBrightness": {"expected": ["augmente", "augmentez", "monte", "rendre plus", "rends plus", "rends les", "augmenter"],
                                "forbidden": ["diminue", "diminuez", "baisse"]},
        "DecreaseBrightness": {"expected": ["diminue", "diminuez", "baisse", "baissez", "attenue", "reduis"],
                                "forbidden": ["augmente", "monte", "allume"]},
        "SetLightBrightness": {"expected": ["regle", "regles", "regler", "ajuste", "ajustez", "mets", "mettre", "fixe", "fixer"],
                                "forbidden": []},
        "SetLightColor":      {"expected": ["change", "changez", "regle", "mets", "mettre", "rends", "rends les", "passe"],
                                "forbidden": []},
    },
}


def _strip_accents(text):
    """Helper used by verb validation; we lowercase and strip common accent marks."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def validate_verb(translation, intent, target_code):
    """Return (ok, reason) where ok=True means the FR text contains an expected verb
    and no forbidden one. If guidance for the (intent, lang) is missing, returns (True, "no-guidance").
    """
    guidance = INTENT_VERB_GUIDANCE.get(target_code, {}).get(intent)
    if not guidance:
        return True, "no-guidance"
    tx = _strip_accents(translation)
    forbidden_hit = [w for w in guidance["forbidden"] if _strip_accents(w) in tx]
    expected_hit = [w for w in guidance["expected"] if _strip_accents(w) in tx]
    if forbidden_hit:
        return False, f"forbidden:{forbidden_hit[0]}"
    if not expected_hit:
        return False, "no-expected-verb"
    return True, expected_hit[0]


def load_slot_translation_dict(config_dir, target_code):
    """Load the slot-value translation dictionary built by build_slot_translations.py.

    Returns a dict {slot_type: {en_value: target_value}}. Returns {} if file missing
    (script falls back to LLM-only translation, with the documented quality risks).
    """
    path = os.path.join(config_dir, f"slot_translations_{target_code}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def translate_command(command, target_language, target_code, slots, intent, slot_dict, model="llama3.3"):
    """Translate a SNIPS smart-lights command to the target language using Ollama.

    Uses the pre-built `slot_dict` to inject the expected slot-value translation
    into the prompt. Also uses INTENT_VERB_GUIDANCE per intent to constrain the
    main verb so the LLM cannot invert semantics (e.g., SwitchLightOn -> "eteins").

    Validates verb in the translation and retries up to 3 times if the verb is
    wrong/forbidden. Returns the best-effort result (with a flag if verb still bad).
    """
    # Build per-slot lookup lines using the dictionary's known translations
    slot_lines = []
    slot_translations = []
    for s in slots:
        st = s.get("slot_type")
        v = s.get("value")
        if not st or not v:
            continue
        v_norm = v.lower().strip()
        expected = slot_dict.get(st, {}).get(v_norm, v)
        slot_lines.append(f'  - {st}: "{v}" -> "{expected}"')
        slot_translations.append({"slot_type": st, "original": v, "translated": expected})
    slot_section = "\n".join(slot_lines) if slot_lines else "  (no slot values)"

    # Verb guidance for this intent in this language
    guidance = INTENT_VERB_GUIDANCE.get(target_code, {}).get(intent, {})
    expected_verbs = guidance.get("expected", [])
    forbidden_verbs = guidance.get("forbidden", [])
    verb_lines = []
    if expected_verbs:
        verb_lines.append(f'  - EXPECTED verbs (use one of): {", ".join(expected_verbs[:5])}')
    if forbidden_verbs:
        verb_lines.append(f'  - FORBIDDEN verbs (do NOT use): {", ".join(forbidden_verbs)}')
    verb_section = "\n".join(verb_lines) if verb_lines else "  (no verb constraint)"

    system_prompt = f"""You are a smart-home (smart-lights) voice command translator.
Translate the following command into {target_language}.

Original command: "{command}"
Intent: "{intent}"
Slot values (use these EXACT translations for slot values; do not substitute synonyms):
{slot_section}
Verb constraints for intent "{intent}" in {target_language}:
{verb_section}

Guidelines for translation:
1. Keep the same intent and meaning as the original command
2. Translate naturally into {target_language} - do NOT do word-by-word translation
3. Keep the translation SHORT and DIRECT - spoken-language style (short, imperative or short question, conversational)
4. Use lowercase text
5. Minimal punctuation
6. For slot values, use the exact translations listed above. The full {target_language} slot phrase MUST appear verbatim in your translation so the slot can be located later.
7. Spell {target_language} verbs and grammar correctly (e.g., for French "dim" use "diminue" or "baisse"; not "dimuie")
8. The MAIN VERB of your translation MUST be from the EXPECTED list above. Do NOT use any FORBIDDEN verbs.

You must return a JSON object with exactly two fields:
- "translation": the translated command string
- "slot_translations": array of objects, one per slot value, each with "slot_type", "original", and "translated" fields

Example for French translation of "turn off the lights in the kitchen":
{{"translation": "eteins les lumieres dans la cuisine", "slot_translations": [{{"slot_type": "room", "original": "kitchen", "translated": "cuisine"}}]}}

Example for French translation of "set the bedroom lights to twenty":
{{"translation": "regle les lumieres de la chambre a vingt", "slot_translations": [{{"slot_type": "room", "original": "bedroom", "translated": "chambre"}}, {{"slot_type": "brightness", "original": "twenty", "translated": "vingt"}}]}}

Return ONLY the JSON object, with no additional text or explanation.
"""

    def _extract_translation(content):
        """Try several strategies to extract the translation string from LLM output."""
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(content)
            if result.get("translation", "").strip():
                return result["translation"]
        except json.JSONDecodeError:
            pass
        json_match = re.search(r'\{[^{}]*"translation"\s*:\s*"[^"]*"[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if result.get("translation", "").strip():
                    return result["translation"]
            except json.JSONDecodeError:
                pass
        trans_match = re.search(r'"translation"\s*:\s*"([^"]+)"', content)
        if trans_match:
            return trans_match.group(1)
        return None

    def _result(translation_text, verb_ok, verb_reason):
        return {
            "translation": translation_text,
            "slot_translations": slot_translations,
            "verb_ok": verb_ok,
            "verb_reason": verb_reason,
        }

    base_user_msg = f"Translate this smart-lights command to {target_language}: {command}"
    last_translation = None
    last_verb_reason = "none"

    # Up to 3 attempts. First attempt with normal prompt. Subsequent attempts add
    # explicit corrective hint if the verb was wrong.
    for attempt in range(3):
        messages = [{"role": "system", "content": system_prompt}]
        if attempt == 0:
            messages.append({"role": "user", "content": base_user_msg})
        else:
            corrective = (
                f"Your previous translation was '{last_translation}'. "
                f"It is invalid because: {last_verb_reason}. "
                f"Generate a NEW translation that uses the EXPECTED verb only. "
                f"Re-translate: {command}"
            )
            messages.append({"role": "user", "content": corrective})

        try:
            response = chat(model=model, messages=messages)
            translation = _extract_translation(response.message.content)
            if not translation:
                continue
            last_translation = translation
            ok, reason = validate_verb(translation, intent, target_code)
            if ok:
                return _result(translation, True, reason)
            last_verb_reason = reason
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue

    # All 3 attempts failed verb check; return last translation flagged
    if last_translation:
        return _result(last_translation, False, last_verb_reason)
    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Translate SNIPS smart-lights commands into target languages")
    parser.add_argument("--input", "-i", type=str,
                        default="data/snips_multilingual_pipeline/snips_commands_v1.json",
                        help="Input JSON file with English commands and BIO tags from step 00")
    parser.add_argument("--output", "-o", type=str,
                        default="data/snips_multilingual_pipeline/snips_translated_all_languages.json",
                        help="Output file path")
    parser.add_argument("--languages", "-l", type=str, nargs="+", default=["French"],
                        help="Target languages for translation (default: French)")
    parser.add_argument("--model", "-m", type=str, default="llama3.3",
                        help="Ollama model to use")
    parser.add_argument("--max-commands", type=int, default=None,
                        help="Maximum number of commands to process (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping already translated commands")
    return parser.parse_args()


def main():
    args = parse_arguments()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    slot_dicts_by_code = {}
    for language in args.languages:
        lc = SNIPS_LANGUAGES.get(language, language[:2].lower())
        d = load_slot_translation_dict(config_dir, lc)
        if d:
            print(f"Loaded slot dictionary for {language} ({lc}): "
                  f"{sum(len(v) for v in d.values())} slot-value entries")
        else:
            print(f"WARNING: no slot dictionary for {language} ({lc}); LLM-only translation will be used")
        slot_dicts_by_code[lc] = d

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            english_data = json.load(f)
        print(f"Loaded {len(english_data)} English commands from {args.input}")
        if args.max_commands and len(english_data) > args.max_commands:
            english_data = english_data[:args.max_commands]
            print(f"Limited to processing {args.max_commands} commands")
    except FileNotFoundError:
        print(f"Input file {args.input} not found!")
        return

    existing_translations = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        for entry in existing_data:
            key = (entry.get("english_command", ""), entry.get("language_code", ""))
            if entry.get("translated_command", "").strip():
                existing_translations[key] = entry
        print(f"Loaded {len(existing_translations)} existing translations for resume")

    multilingual_data = []
    for row in tqdm(english_data, desc="Translating commands"):
        english_command = row["command"]
        intent = row.get("intent", "")
        slots = row.get("slots", [])
        bio_tags = row.get("bio_tags", "")
        tokens = row.get("tokens", [])

        # English entry (identity translation)
        english_entry = {
            "english_command": english_command,
            "language": "English",
            "language_code": "en",
            "translated_command": english_command,
            "intent": intent,
            "slots": slots,
            "slot_translations": [
                {"slot_type": s["slot_type"], "original": s["value"], "translated": s["value"]}
                for s in slots if s.get("slot_type") and s.get("value")
            ],
            "bio_tags": bio_tags,
            "tokens": tokens,
            "token_count": len(tokens),
            "category": row.get("category", ""),
        }
        multilingual_data.append(english_entry)

        # Each target language
        for language in args.languages:
            lang_code = SNIPS_LANGUAGES.get(language, language[:2].lower())

            key = (english_command, lang_code)
            if key in existing_translations:
                multilingual_data.append(existing_translations[key])
                continue

            result = translate_command(
                english_command,
                target_language=language,
                target_code=lang_code,
                slots=slots,
                intent=intent,
                slot_dict=slot_dicts_by_code.get(lang_code, {}),
                model=args.model,
            )

            if result and "translation" in result:
                multilingual_entry = {
                    "english_command": english_command,
                    "language": language,
                    "language_code": lang_code,
                    "translated_command": result["translation"],
                    "intent": intent,
                    "slots": slots,
                    "slot_translations": result.get("slot_translations", []),
                    "bio_tags": "",
                    "tokens": [],
                    "token_count": 0,
                    "category": row.get("category", ""),
                    "verb_ok": result.get("verb_ok", True),
                    "verb_reason": result.get("verb_reason", "n/a"),
                }
                multilingual_data.append(multilingual_entry)
            else:
                multilingual_data.append({
                    "english_command": english_command,
                    "language": language,
                    "language_code": lang_code,
                    "translated_command": "",
                    "intent": intent,
                    "slots": slots,
                    "slot_translations": [],
                    "bio_tags": "",
                    "tokens": [],
                    "token_count": 0,
                    "category": row.get("category", ""),
                })
                print(f"  Warning: Failed to translate to {language}, added empty entry")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(multilingual_data, f, ensure_ascii=False, indent=2)

    print(f"\nMultilingual translations saved to {args.output}")
    print(f"Total entries: {len(multilingual_data)}")
    print(f"Languages: ['English'] + {args.languages}")
    print(f"English commands: {len(english_data)}")

    language_counts = {}
    empty_counts = {}
    for entry in multilingual_data:
        lang = entry["language"]
        language_counts[lang] = language_counts.get(lang, 0) + 1
        if not entry.get("translated_command", "").strip():
            empty_counts[lang] = empty_counts.get(lang, 0) + 1

    print("\nLanguage breakdown:")
    for lang, count in language_counts.items():
        empty = empty_counts.get(lang, 0)
        suffix = f" ({empty} empty)" if empty > 0 else ""
        print(f"  {lang}: {count} entries{suffix}")

    # Verb-validation stats per non-English language
    print("\nVerb validation stats (per target language):")
    from collections import Counter
    verb_fail_by_lang_intent = {}
    for entry in multilingual_data:
        if entry.get("language") == "English":
            continue
        if entry.get("verb_ok", True):
            continue
        key = (entry["language"], entry["intent"])
        verb_fail_by_lang_intent.setdefault(key, Counter())[entry.get("verb_reason", "?")] += 1
    if not verb_fail_by_lang_intent:
        print("  All translations passed verb validation.")
    else:
        for (lang, intent), counter in sorted(verb_fail_by_lang_intent.items()):
            total = sum(counter.values())
            print(f"  {lang} / {intent}: {total} entries failed verb check -> {dict(counter)}")


if __name__ == "__main__":
    main()
