"""
Generate monolingual synthetic banking commands for the skit-s2i dataset
using the seed-pattern approach.

The skit-s2i dataset has:
- 14 intents (Banking domain)
- ~11,845 total samples (10,445 train + 1,400 test)
- Intent-only classification (no entity slots)
- English language only

Seed-pattern approach:
- Original Skit-S2I templates are loaded from skit-s2i/data/*.parquet
- LLM is anchored to those templates as the *only* valid intent meaning
- LLM only produces surface-form variations of those templates
- Anti-confusion: LLM is shown the full intent ontology and instructed
  to discard variations that could be interpreted as a different intent

This addresses the 31.7% cross-intent confusion observed in the previous run.

Usage:
    python generate_monolingual_SISE_data.py --num-examples 850 --batch-size 50 \
        --output banking_commands_dataset.csv \
        --parquet-dir ../../skit-s2i/data
"""

import csv
import glob
import json
import argparse
import os
import time
from collections import defaultdict
from ollama import chat
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    pd = None


# Define the 14 banking intents from skit-s2i dataset.
# Each intent has no entity slots (intent-only classification).
# Descriptions are paraphrased from the original skit-s2i/intent_info.csv.
intents = [
    {
        "intent": "branch_address",
        "description": "Asking for bank branch location, address, or directions to a branch",
    },
    {
        "intent": "activate_card",
        "description": "Requesting to activate a new debit or credit card",
    },
    {
        "intent": "past_transactions",
        "description": "Inquiring about previous transactions, transaction history, or past account activity in a specific time period",
    },
    {
        "intent": "dispatch_status",
        "description": "Checking dispatch / delivery status of card products (debit card, credit card)",
    },
    {
        "intent": "outstanding_balance",
        "description": "Asking about outstanding dues or pending amount on card products (typically credit card)",
    },
    {
        "intent": "card_issue",
        "description": "Reporting a problem with using a card product (card not working, declined transactions, payment failure)",
    },
    {
        "intent": "ifsc_code",
        "description": "Asking for the IFSC code of a bank branch for fund transfers",
    },
    {
        "intent": "generate_pin",
        "description": "Requesting to generate or change the PIN for a card product (debit / credit / ATM PIN)",
    },
    {
        "intent": "unauthorised_transaction",
        "description": "Reporting an unauthorised or fraudulent transaction on the account",
    },
    {
        "intent": "loan_query",
        "description": "General queries about different kinds of loans (home loan, personal loan, eligibility, application)",
    },
    {
        "intent": "balance_enquiry",
        "description": "Checking current bank account balance (savings / current account)",
    },
    {
        "intent": "change_limit",
        "description": "Requesting to change the transaction / withdrawal / spend limit on a card product",
    },
    {
        "intent": "block",
        "description": "Requesting to block a card or banking product",
    },
    {
        "intent": "lost",
        "description": "Reporting a lost card product",
    },
]

# Intent categories for banking domain
intent_categories = {
    "branch_address": "Branch Information",
    "activate_card": "Card Services",
    "past_transactions": "Account Information",
    "dispatch_status": "Card Services",
    "outstanding_balance": "Account Information",
    "card_issue": "Card Services",
    "ifsc_code": "Branch Information",
    "generate_pin": "Card Services",
    "unauthorised_transaction": "Fraud & Security",
    "loan_query": "Loan Services",
    "balance_enquiry": "Account Information",
    "change_limit": "Card Services",
    "block": "Card Services",
    "lost": "Card Services",
}

# Map skit-s2i intent_class index -> intent name (mirrors skit-s2i/intent_info.csv)
INTENT_CLASS_TO_NAME = {
    0: "branch_address",
    1: "activate_card",
    2: "past_transactions",
    3: "dispatch_status",
    4: "outstanding_balance",
    5: "card_issue",
    6: "ifsc_code",
    7: "generate_pin",
    8: "unauthorised_transaction",
    9: "loan_query",
    10: "balance_enquiry",
    11: "change_limit",
    12: "block",
    13: "lost",
}


def load_seed_templates_from_parquet(parquet_dir):
    """Read original skit-s2i parquet files and return {intent_name: [template, ...]}.

    Templates are the literal `template` field used by the original speakers as guides.
    They are intentionally short and intent-specific; placeholders like <numeric>,
    <Month>, (credit/debit), {{Axis Bank}} are kept as-is and expanded by the LLM
    via the prompt rules.
    """
    if pd is None:
        raise RuntimeError("pandas is required to load seed templates from parquet files")
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {parquet_dir}. "
            f"Pass --parquet-dir to point at skit-s2i/data/"
        )
    dfs = [pd.read_parquet(f, columns=["intent_class", "template"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    seeds = defaultdict(list)
    for ic, t in zip(df["intent_class"], df["template"]):
        name = INTENT_CLASS_TO_NAME.get(int(ic))
        if not name:
            continue
        # Normalise whitespace, drop trivial duplicates
        t_clean = str(t).strip()
        if t_clean and t_clean not in seeds[name]:
            seeds[name].append(t_clean)
    return dict(seeds)


def generate_examples(
    intent_info,
    seed_templates,
    other_intents,
    num_examples=50,
    model="llama3.2",
    language="English",
    existing_commands=None,
):
    """Generate examples for the given intent using Ollama with the seed-pattern prompt.

    Args:
        intent_info: dict with `intent` and `description`
        seed_templates: list of original templates that anchor the intent meaning
        other_intents: list of (intent_name, description) tuples for the other 13 intents
        num_examples: target number of variations to produce in this call
        model: Ollama model name
        language: language for generation (kept English for skit-s2i)
        existing_commands: list of already-generated commands to discourage repeats
    """

    existing_text = ""
    if existing_commands and len(existing_commands) > 0:
        sample_existing = existing_commands[-20:]
        existing_text = (
            "\n\nPreviously generated examples (DO NOT repeat these):\n"
            + "\n".join([f"- {cmd}" for cmd in sample_existing])
        )

    seed_block = "\n".join([f"- {t}" for t in seed_templates])
    other_intents_block = "\n".join(
        [f"- {n}: {d}" for n, d in other_intents]
    )

    # Prompt follows the senior's monolingual prompt structure (numbered guidelines,
    # JSON output, good/bad examples). Specialised to banking + seed-pattern.
    system_prompt = f"""You are a banking voice command dataset generator for Indian banking customers.
Generate {num_examples} realistic voice commands in {language} with the intent "{intent_info['intent']}".
Description of the intent: {intent_info['description']}

Seed templates (generate ONLY surface-form variations of THESE; do not invent new intent meaning):
{seed_block}

Other banking intents in this dataset (your variation MUST NOT match any of these meanings):
{other_intents_block}

Each command should:
1. Be a natural banking query or request a customer would say on a phone call to a bank
2. Be a surface-form variation of one of the seed templates above (paraphrase, reorder, add filler words, change formality)
3. Stay strictly within the meaning of the "{intent_info['intent']}" intent. If your variation could be interpreted as ANY other intent from the list above, discard it and try again
4. Vary in phrasing, length, and formality. Include both very short queries (1-3 words, like some seed templates) and longer descriptive ones, so the length distribution stays natural
5. Use Indian English expressions where appropriate (e.g., "kindly", "please do the needful")
6. If a seed template contains a placeholder, substitute a realistic value:
   - <numeric> -> a number such as 3, 5, last 10
   - <Month> -> a real month such as January, March, last month
   - (credit/debit) -> pick one (credit OR debit)
   - {{{{Axis Bank}}}} or any {{{{bank}}}} -> an Indian bank name (HDFC, ICICI, SBI, Axis, Kotak)

Style guidelines:
- Banking customers naturally ask questions ("What is my balance?") and make requests ("Block my card immediately"). Both are acceptable.
- Use Indian context (Indian English, Indian bank names, INR if amounts are mentioned)
- Avoid explicit content or inappropriate language
- Avoid using any keyword that primarily belongs to a different intent (e.g., do NOT mention "balance" in a branch_address variation)
{existing_text}

Format your response as a JSON list of objects, with each object having:
- "command": The full voice command

Example format:
[
  {{"command": "Bank address"}},
  {{"command": "Where is the nearest HDFC branch located?"}}
]

DO NOT use commands like:
- "What is my account balance?" (this belongs to the balance_enquiry intent, not "{intent_info['intent']}")
- "I want to block my card" (this belongs to the block intent, not "{intent_info['intent']}")

Return ONLY the JSON list, with no additional text.
"""

    messages = [{"role": "user", "content": system_prompt}]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chat(model=model, messages=messages)
            content = response.message.content

            try:
                examples = json.loads(content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r"\[[\s\S]*\]", content)
                if json_match:
                    try:
                        examples = json.loads(json_match.group())
                    except json.JSONDecodeError as e:
                        print(f"Error decoding response for {intent_info['intent']}: {e}")
                        print(f"Response content: {content[:500]}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        return []
                else:
                    print(f"No JSON array found in response for {intent_info['intent']}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return []

            result = []
            for ex in examples:
                if isinstance(ex, dict) and "command" in ex:
                    cmd = ex["command"].strip()
                    if len(cmd) > 1 and cmd not in (existing_commands or []):
                        result.append({
                            "command": cmd,
                            "intent": intent_info["intent"],
                            "category": intent_categories.get(intent_info["intent"], "Banking"),
                        })
            return result

        except Exception as e:
            print(f"Error generating examples for {intent_info['intent']}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return []

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Generate banking voice commands for skit-s2i dataset (seed-pattern approach)"
    )
    parser.add_argument("--output", type=str, default="banking_commands_dataset.csv",
                        help="Output CSV file path")
    parser.add_argument("--num-examples", type=int, default=850,
                        help="Total examples per intent (default: 850, ~11,900 total)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Examples to generate per LLM call (default: 50)")
    parser.add_argument("--model", type=str, default="llama3.2",
                        help="Ollama model to use (default: llama3.2)")
    parser.add_argument("--language", type=str, default="English",
                        help="Language for commands (default: English)")
    parser.add_argument("--intent", type=str, default=None,
                        help="Generate for a specific intent only (optional, used for the pilot run)")
    parser.add_argument("--parquet-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "..", "skit-s2i", "data"),
                        help="Path to original skit-s2i parquet files (for seed templates)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file if it exists")
    parser.add_argument("--max-iterations", type=int, default=100,
                        help="Maximum LLM batches per intent (safeguard against infinite loops)")

    args = parser.parse_args()

    # Load seed templates from the original parquet files
    print(f"Loading seed templates from: {args.parquet_dir}")
    seed_templates_by_intent = load_seed_templates_from_parquet(args.parquet_dir)
    for n in [i["intent"] for i in intents]:
        if n not in seed_templates_by_intent:
            print(f"  WARNING: no seed templates found for intent {n}")
        else:
            print(f"  {n}: {len(seed_templates_by_intent[n])} seed templates loaded")

    # Choose intents to process
    if args.intent:
        intents_to_process = [i for i in intents if i["intent"] == args.intent]
        if not intents_to_process:
            print(f"Intent '{args.intent}' not found. Available intents:")
            for i in intents:
                print(f"  - {i['intent']}")
            return
    else:
        intents_to_process = intents

    print(f"Generating {args.num_examples} examples per intent for {len(intents_to_process)} intents...")
    print(f"Total target: ~{args.num_examples * len(intents_to_process)} examples")

    # Load existing data if resuming
    existing_data = {}
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from existing file: {args.output}")
        with open(args.output, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                intent = row["intent"]
                existing_data.setdefault(intent, []).append(row["command"])
        print(f"Loaded {sum(len(v) for v in existing_data.values())} existing examples")

    all_examples = []

    for intent_info in tqdm(intents_to_process, desc="Generating examples for intents"):
        intent_name = intent_info["intent"]
        category = intent_categories.get(intent_name, "Banking")

        seed_templates = seed_templates_by_intent.get(intent_name, [])
        if not seed_templates:
            print(f"\nSkipping {intent_name}: no seed templates available")
            continue

        # other_intents = the other 13 intent_name/description pairs (anti-confusion)
        other_intents = [
            (i["intent"], i["description"])
            for i in intents if i["intent"] != intent_name
        ]

        existing_commands = existing_data.get(intent_name, [])
        examples_needed = args.num_examples - len(existing_commands)

        if examples_needed <= 0:
            print(f"Already have {len(existing_commands)} examples for {intent_name}, skipping...")
            for cmd in existing_commands:
                all_examples.append({"command": cmd, "intent": intent_name, "category": category})
            continue

        print(f"\nGenerating {examples_needed} examples for {intent_name} (Category: {category})...")
        print(f"  Using {len(seed_templates)} seed templates")

        intent_examples = []
        collected_commands = list(existing_commands)
        max_iterations = args.max_iterations
        iteration = 0

        with tqdm(total=examples_needed, desc=f"  Generating for {intent_name}", leave=False) as pbar:
            while len(intent_examples) < examples_needed and iteration < max_iterations:
                iteration += 1
                remaining = examples_needed - len(intent_examples)
                batch_target = min(args.batch_size, remaining + 20)

                examples = generate_examples(
                    intent_info,
                    seed_templates=seed_templates,
                    other_intents=other_intents,
                    num_examples=batch_target,
                    model=args.model,
                    language=args.language,
                    existing_commands=collected_commands,
                )

                added_count = 0
                for ex in examples:
                    if ex["command"] not in collected_commands and len(intent_examples) < examples_needed:
                        intent_examples.append(ex)
                        collected_commands.append(ex["command"])
                        added_count += 1

                pbar.update(added_count)
                time.sleep(0.5)

        for cmd in existing_commands:
            all_examples.append({"command": cmd, "intent": intent_name, "category": category})
        all_examples.extend(intent_examples)

        print(f"  Collected {len(intent_examples)} new + {len(existing_commands)} existing = "
              f"{len(intent_examples) + len(existing_commands)} total for {intent_name}")

    print(f"\nWriting {len(all_examples)} examples to {args.output}...")
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["command", "intent", "category"])
        writer.writeheader()
        for ex in all_examples:
            writer.writerow(ex)

    print("\nGeneration Statistics")
    intent_counts = defaultdict(int)
    for ex in all_examples:
        intent_counts[ex["intent"]] += 1
    print(f"Total examples: {len(all_examples)}")
    print("\nPer intent breakdown:")
    for intent, count in sorted(intent_counts.items()):
        category = intent_categories.get(intent, "Banking")
        print(f"  {intent}: {count} examples (Category: {category})")
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
