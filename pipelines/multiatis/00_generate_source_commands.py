#!/usr/bin/env python3
"""
Generate ATIS-style English source commands with full 84-slot BIO annotations.
Uses template-based generation with value pools to ensure all slot types are covered
and the intent distribution matches the original ATIS dataset.

Output: multiatis_commands_v3.json
"""

import json
import os
import re
import random
import argparse
from collections import Counter
from tqdm import tqdm

def load_taxonomy(config_path="config/atis_slot_taxonomy.json"):
    """Load slot taxonomy configuration."""
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ── Slot-type to value-pool key mapping ──────────────────────────
SLOT_TO_POOL = {
    "fromloc.city_name": "city", "toloc.city_name": "city",
    "stoploc.city_name": "city", "city_name": "city",
    "fromloc.airport_name": "airport", "toloc.airport_name": "airport",
    "stoploc.airport_name": "airport", "airport_name": "airport",
    "fromloc.airport_code": "airport_code", "toloc.airport_code": "airport_code",
    "stoploc.airport_code": "airport_code", "airport_code": "airport_code",
    "fromloc.state_name": "state", "toloc.state_name": "state",
    "stoploc.state_code": "state_code", "state_name": "state",
    "fromloc.state_code": "state_code", "toloc.state_code": "state_code",
    "state_code": "state_code", "toloc.country_name": "country",
    "airline_name": "airline", "airline_code": "airline_code",
    "aircraft_code": "aircraft_code", "flight_number": "flight_number",
    "depart_date.day_name": "day_name", "arrive_date.day_name": "day_name",
    "return_date.day_name": "day_name", "day_name": "day_name",
    "depart_date.month_name": "month_name", "arrive_date.month_name": "month_name",
    "return_date.month_name": "month_name", "month_name": "month_name",
    "depart_date.day_number": "day_number", "arrive_date.day_number": "day_number",
    "return_date.day_number": "day_number", "day_number": "day_number",
    "depart_date.date_relative": "date_relative", "arrive_date.date_relative": "date_relative",
    "return_date.date_relative": "date_relative",
    "depart_date.today_relative": "today_relative", "arrive_date.today_relative": "today_relative",
    "return_date.today_relative": "today_relative", "today_relative": "today_relative",
    "depart_date.year": "year", "arrive_date.year": "year",
    "depart_time.time": "time", "arrive_time.time": "time", "time": "time",
    "depart_time.start_time": "time", "arrive_time.start_time": "time",
    "depart_time.end_time": "time", "arrive_time.end_time": "time",
    "depart_time.period_of_day": "period_of_day", "arrive_time.period_of_day": "period_of_day",
    "period_of_day": "period_of_day", "return_time.period_of_day": "period_of_day",
    "depart_time.time_relative": "time_relative", "arrive_time.time_relative": "time_relative",
    "time_relative": "time_relative",
    "depart_time.period_mod": "period_mod", "arrive_time.period_mod": "period_mod",
    "return_time.period_mod": "period_mod",
    "class_type": "class_type", "flight_mod": "flight_mod",
    "transport_type": "transport_type", "meal": "meal",
    "meal_code": "meal_code", "meal_description": "meal_description",
    "cost_relative": "cost_relative", "fare_amount": "fare_amount",
    "fare_basis_code": "fare_basis_code", "restriction_code": "restriction_code",
    "flight_days": "flight_days", "connect": "connect",
    "round_trip": "round_trip", "economy": "economy",
    "days_code": "days_code", "flight_stop": "flight_stop",
    "flight_time": "flight_time", "mod": "mod", "or": "or",
}

# ── Templates per intent ─────────────────────────────────────────
# {slot_type} placeholders are replaced with random values from pools
# Multiple templates per intent for variety

TEMPLATES = {
    "flight": [
        "show me flights from {fromloc.city_name} to {toloc.city_name}",
        "i need a flight from {fromloc.city_name} to {toloc.city_name}",
        "what flights go from {fromloc.city_name} to {toloc.city_name}",
        "list flights from {fromloc.city_name} to {toloc.city_name}",
        "flights from {fromloc.city_name} to {toloc.city_name}",
        "show flights from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name}",
        "i want a flight from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name}",
        "flights from {fromloc.city_name} to {toloc.city_name} in the {depart_time.period_of_day}",
        "show me {flight_mod} flights from {fromloc.city_name} to {toloc.city_name}",
        "i need a {flight_mod} flight from {fromloc.city_name} to {toloc.city_name}",
        "what are the flights from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name} {depart_time.period_of_day}",
        "find a flight from {fromloc.city_name} to {toloc.city_name} departing {depart_date.today_relative}",
        "are there any flights from {fromloc.city_name} to {toloc.city_name} {depart_date.today_relative}",
        "show {airline_name} flights from {fromloc.city_name} to {toloc.city_name}",
        "i want to fly from {fromloc.city_name} to {toloc.city_name} on {depart_date.month_name} {depart_date.day_number}",
        "flights from {fromloc.city_name} to {toloc.city_name} leaving {depart_time.period_of_day}",
        "what flights leave {fromloc.city_name} for {toloc.city_name} after {depart_time.time}",
        "show me flights from {fromloc.city_name} to {toloc.city_name} before {arrive_time.time}",
        "i need a {class_type} flight from {fromloc.city_name} to {toloc.city_name}",
        "show {round_trip} flights from {fromloc.city_name} to {toloc.city_name}",
        "flights from {fromloc.city_name} to {toloc.city_name} with {flight_stop}",
        "what {flight_mod} flights are there from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name}",
        "list {airline_name} flights from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name}",
        "show flights from {fromloc.city_name} to {toloc.city_name} arriving before {arrive_time.time}",
        "i want a flight from {fromloc.airport_name} to {toloc.city_name}",
        "flights from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name} after {depart_time.time}",
        "show flights from {fromloc.city_name} to {toloc.city_name} {depart_date.today_relative} in the {depart_time.period_of_day}",
        "what flights from {fromloc.city_name} to {toloc.city_name} arrive in the {arrive_time.period_of_day}",
        "show me flights from {fromloc.city_name} to {toloc.city_name} stopping in {stoploc.city_name}",
        "i need a {class_type} {flight_mod} flight from {fromloc.city_name} to {toloc.city_name}",
        "flights on {airline_name} from {fromloc.city_name} to {toloc.city_name} in {depart_date.month_name}",
        "what is the earliest flight from {fromloc.city_name} to {toloc.city_name}",
        "show me the latest flight from {fromloc.city_name} to {toloc.city_name}",
        "are there {depart_time.period_of_day} flights from {fromloc.city_name} to {toloc.city_name}",
        "flights from {fromloc.city_name} to {toloc.city_name} returning on {return_date.day_name}",
        "i need a flight from {fromloc.state_name} to {toloc.city_name}",
        "show flights departing {fromloc.city_name} arriving {toloc.city_name} by {arrive_time.time}",
        "find {flight_mod} flights from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name} {depart_time.period_of_day}",
        # ── Templates for missing slot types ──
        "flights from {fromloc.airport_code} to {toloc.airport_code}",
        "show flights from {fromloc.city_name} to {toloc.airport_name}",
        "flights from {fromloc.state_code} to {toloc.state_name}",
        "show flights from {fromloc.city_name} to {toloc.country_name}",
        "flights from {fromloc.city_name} to {toloc.state_code} on {depart_date.day_name}",
        "flights from {fromloc.city_name} to {toloc.city_name} arriving on {arrive_date.day_name}",
        "flights from {fromloc.city_name} to {toloc.city_name} arriving {arrive_date.month_name} {arrive_date.day_number}",
        "flights from {fromloc.city_name} to {toloc.city_name} arriving {arrive_date.today_relative}",
        "flights from {fromloc.city_name} to {toloc.city_name} arriving {arrive_date.date_relative}",
        "flights from {fromloc.city_name} to {toloc.city_name} departing between {depart_time.start_time} and {depart_time.end_time}",
        "flights from {fromloc.city_name} to {toloc.city_name} arriving between {arrive_time.start_time} and {arrive_time.end_time}",
        "flights from {fromloc.city_name} to {toloc.city_name} departing {depart_time.time_relative}",
        "flights from {fromloc.city_name} to {toloc.city_name} arriving {arrive_time.time_relative}",
        "flights from {fromloc.city_name} to {toloc.city_name} {depart_time.period_mod} {depart_time.period_of_day}",
        "flights from {fromloc.city_name} to {toloc.city_name} {arrive_time.period_mod} {arrive_time.period_of_day}",
        "flights from {fromloc.city_name} to {toloc.city_name} {depart_date.date_relative}",
        "flights from {fromloc.city_name} to {toloc.city_name} in {depart_date.year}",
        "flights from {fromloc.city_name} to {toloc.city_name} with {connect}",
        "flights from {fromloc.city_name} to {toloc.city_name} {flight_days}",
        "show {flight_days} flights from {fromloc.city_name} to {toloc.city_name}",
        "flights from {fromloc.city_name} to {toloc.city_name} returning {return_date.today_relative}",
        "flights from {fromloc.city_name} to {toloc.city_name} returning in {return_date.month_name}",
        "flights from {fromloc.city_name} to {toloc.city_name} returning on {return_date.month_name} {return_date.day_number}",
        "flights from {fromloc.city_name} to {toloc.city_name} return {return_date.date_relative}",
        "flights from {fromloc.city_name} to {toloc.city_name} at {time}",
        "flights from {fromloc.city_name} to {toloc.city_name} stopping at {stoploc.airport_name}",
        "flights from {fromloc.city_name} to {toloc.city_name} via {stoploc.state_code}",
        "flights from {fromloc.city_name} {or} {toloc.city_name} to {stoploc.city_name}",
        "show {mod} expensive flights from {fromloc.city_name} to {toloc.city_name}",
        "flights on {day_name} from {fromloc.city_name} to {toloc.city_name}",
        "flights on {month_name} {day_number} from {fromloc.city_name} to {toloc.city_name}",
        "flights {today_relative} from {fromloc.city_name} to {toloc.city_name}",
        "flights in the {period_of_day} from {fromloc.city_name} to {toloc.city_name}",
        "flights {time_relative} from {fromloc.city_name} to {toloc.city_name}",
        "flights under {flight_time} from {fromloc.city_name} to {toloc.city_name}",
        "show flights from {fromloc.city_name} to {toloc.city_name} on {days_code}",
        "flights returning {return_time.period_of_day} from {toloc.city_name} to {fromloc.city_name}",
        "flights returning {return_time.period_mod} from {toloc.city_name} to {fromloc.city_name}",
    ],
    "airfare": [
        "what is the fare from {fromloc.city_name} to {toloc.city_name}",
        "how much is a flight from {fromloc.city_name} to {toloc.city_name}",
        "show me the fares from {fromloc.city_name} to {toloc.city_name}",
        "airfare from {fromloc.city_name} to {toloc.city_name}",
        "what does a {round_trip} ticket from {fromloc.city_name} to {toloc.city_name} cost",
        "show {class_type} fares from {fromloc.city_name} to {toloc.city_name}",
        "what is the {cost_relative} fare from {fromloc.city_name} to {toloc.city_name}",
        "fares for flights from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name}",
        "how much does a {class_type} ticket from {fromloc.city_name} to {toloc.city_name} cost",
        "show me fares from {fromloc.city_name} to {toloc.city_name} on {airline_name}",
        "what is the fare for a {round_trip} flight from {fromloc.city_name} to {toloc.city_name}",
        "list fares under {fare_amount} from {fromloc.city_name} to {toloc.city_name}",
        "what is the {economy} fare from {fromloc.city_name} to {toloc.city_name}",
        "show me fare code {fare_basis_code} from {fromloc.city_name} to {toloc.city_name}",
    ],
    "ground_service": [
        "what ground transportation is available in {fromloc.city_name}",
        "show me ground transportation in {fromloc.city_name}",
        "is there {transport_type} service in {fromloc.city_name}",
        "i need {transport_type} service from {fromloc.city_name} to {toloc.city_name}",
        "what {transport_type} services are available in {fromloc.city_name}",
        "ground transportation from {fromloc.city_name} airport to downtown",
        "is there a {transport_type} from {fromloc.city_name} airport",
        "show ground services in {city_name}",
        "what is the ground transportation in {city_name}",
        "i need ground transportation in {fromloc.city_name}",
    ],
    "airline": [
        "what airlines fly from {fromloc.city_name} to {toloc.city_name}",
        "which airlines serve {fromloc.city_name}",
        "show me airlines that fly from {fromloc.city_name} to {toloc.city_name}",
        "does {airline_name} fly from {fromloc.city_name} to {toloc.city_name}",
        "what airline is {airline_code}",
        "tell me about {airline_name}",
        "which airlines go from {fromloc.city_name} to {toloc.city_name}",
        "list airlines serving {fromloc.city_name} to {toloc.city_name}",
        "airlines from {fromloc.city_name} to {toloc.city_name} in the {depart_time.period_of_day}",
    ],
    "abbreviation": [
        "what does {airline_code} stand for",
        "what is the meaning of {airline_code}",
        "what does {airport_code} mean",
        "what is {fare_basis_code}",
        "explain the code {restriction_code}",
        "what does the abbreviation {airline_code} mean",
        "what is the abbreviation {airport_code}",
        "define {fare_basis_code}",
        "what does code {airline_code} represent",
    ],
    "aircraft": [
        "what type of aircraft is used on flights from {fromloc.city_name} to {toloc.city_name}",
        "what aircraft does {airline_name} use from {fromloc.city_name} to {toloc.city_name}",
        "show me the aircraft for flight {flight_number}",
        "what kind of plane is a {aircraft_code}",
        "what aircraft is used on {airline_name} flight {flight_number}",
        "describe aircraft {aircraft_code}",
        "what type of aircraft is {aircraft_code}",
    ],
    "flight_time": [
        "how long is the flight from {fromloc.city_name} to {toloc.city_name}",
        "what is the flight time from {fromloc.city_name} to {toloc.city_name}",
        "how many hours from {fromloc.city_name} to {toloc.city_name}",
        "flight duration from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name}",
        "how long does it take to fly from {fromloc.city_name} to {toloc.city_name}",
        "what is the duration of {airline_name} flights from {fromloc.city_name} to {toloc.city_name}",
    ],
    "quantity": [
        "how many flights from {fromloc.city_name} to {toloc.city_name}",
        "how many flights does {airline_name} have from {fromloc.city_name} to {toloc.city_name}",
        "how many {depart_time.period_of_day} flights from {fromloc.city_name} to {toloc.city_name}",
        "how many flights go from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name}",
        "what is the number of flights from {fromloc.city_name} to {toloc.city_name}",
    ],
    "city": [
        "what cities does {airline_name} serve",
        "show me cities served by {airline_name}",
        "what cities are served from {city_name}",
        "list cities in {state_name}",
        "what city is {airport_name} in",
    ],
    "ground_fare": [
        "what is the cost of {transport_type} in {fromloc.city_name}",
        "how much is {transport_type} service from {fromloc.city_name}",
        "ground transportation fares in {fromloc.city_name}",
        "what does a {transport_type} cost in {fromloc.city_name}",
        "show {transport_type} fares in {fromloc.city_name}",
    ],
    "distance": [
        "how far is {fromloc.city_name} from {toloc.city_name}",
        "what is the distance from {fromloc.city_name} to {toloc.city_name}",
        "distance between {fromloc.city_name} and {toloc.city_name}",
        "how many miles from {fromloc.city_name} to {toloc.city_name}",
    ],

    "capacity": [
        "what is the seating capacity of a {aircraft_code}",
        "how many seats on a {aircraft_code}",
        "capacity of aircraft {aircraft_code}",
        "what is the capacity on {airline_name} flights from {fromloc.city_name} to {toloc.city_name}",
    ],
    "flight_no": [
        "what is the flight number for {airline_name} from {fromloc.city_name} to {toloc.city_name}",
        "show me {airline_name} flight numbers from {fromloc.city_name} to {toloc.city_name}",
        "what flight number is {airline_name} flight {flight_number}",
    ],
    "meal": [
        "what meals are served on flights from {fromloc.city_name} to {toloc.city_name}",
        "is {meal} served on the flight from {fromloc.city_name} to {toloc.city_name}",
        "what meal is on {airline_name} flight {flight_number}",
        "do flights from {fromloc.city_name} to {toloc.city_name} serve {meal}",
        "what is meal code {meal_code}",
        "describe the {meal_description} on flights from {fromloc.city_name} to {toloc.city_name}",
    ],
    "restriction": [
        "what is restriction {restriction_code}",
        "explain restriction code {restriction_code}",
        "what does restriction {restriction_code} mean",
        "show me the restrictions for fare code {fare_basis_code}",
    ],
    "cheapest": [
        "what is the cheapest flight from {fromloc.city_name} to {toloc.city_name}",
        "show me the cheapest airfare from {fromloc.city_name} to {toloc.city_name}",
        "cheapest fare from {fromloc.city_name} to {toloc.city_name} on {depart_date.day_name}",
        "find the cheapest {round_trip} flight from {fromloc.city_name} to {toloc.city_name}",
    ],
    "day_name": [
        "what days does {airline_name} fly from {fromloc.city_name} to {toloc.city_name}",
        "on what days are there flights from {fromloc.city_name} to {toloc.city_name}",
        "which {depart_date.day_name} flights go from {fromloc.city_name} to {toloc.city_name}",
    ],
    "airport": [
        "what airports are in {city_name}",
        "show me airports in {city_name}",
        "which airport serves {city_name}",
        "what is the airport code for {airport_name}",
        "list airports in {state_name}",
        "what airport is closest to {city_name}",
        "show airports in {state_code}",
    ],
}

def get_pool_value(slot_type, pools, used_values=None):
    """Get a random value from the appropriate pool for a slot type."""
    pool_key = SLOT_TO_POOL.get(slot_type)
    if not pool_key or pool_key not in pools:
        # Handle special cases
        if slot_type == "or":
            return "or"
        if "year" in slot_type:
            return random.choice(["1993", "1994", "1995", "1996"])
        return None

    pool = pools[pool_key]
    if used_values:
        available = [v for v in pool if v not in used_values]
        if available:
            return random.choice(available)
    return random.choice(pool)

def create_bio_tags(command_text, slots):
    """Create BIO tags by finding slot values in the tokenized command."""
    tokens = command_text.lower().split()
    bio_tags = ['O'] * len(tokens)

    # Sort slots by position in text (leftmost first), longer values first for overlap handling
    slot_matches = []
    for slot in slots:
        value = slot['value'].lower()
        value_tokens = value.split()
        # Find position in token list
        for i in range(len(tokens) - len(value_tokens) + 1):
            if tokens[i:i + len(value_tokens)] == value_tokens:
                slot_matches.append((i, len(value_tokens), slot['slot_type']))
                break

    # Sort by position, apply tags (no overlaps)
    slot_matches.sort(key=lambda x: x[0])
    tagged = set()
    for start, length, slot_type in slot_matches:
        positions = set(range(start, start + length))
        if positions & tagged:
            continue  # skip overlapping
        bio_tags[start] = f'B-{slot_type}'
        for j in range(1, length):
            bio_tags[start + j] = f'I-{slot_type}'
        tagged.update(positions)

    return tokens, bio_tags

def generate_command_from_template(template, intent, pools):
    """Generate a single command from a template with random slot values."""
    # Find all slot placeholders in template
    slot_pattern = re.compile(r'\{([^}]+)\}')
    slot_types = slot_pattern.findall(template)

    # Generate unique values for each slot
    used_cities = set()
    slots = []
    filled_template = template

    for slot_type in slot_types:
        # Ensure from/to cities are different
        if 'city_name' in slot_type:
            value = get_pool_value(slot_type, pools, used_cities)
            if value:
                used_cities.add(value)
        else:
            value = get_pool_value(slot_type, pools)

        if value is None:
            value = "unknown"

        slots.append({"slot_type": slot_type, "value": value})
        filled_template = filled_template.replace('{' + slot_type + '}', value, 1)

    # Create BIO tags
    tokens, bio_tags = create_bio_tags(filled_template, slots)

    # Compute slot spans
    final_slots = []
    for slot in slots:
        value_tokens = slot['value'].lower().split()
        for i in range(len(tokens) - len(value_tokens) + 1):
            if tokens[i:i + len(value_tokens)] == value_tokens:
                final_slots.append({
                    "slot_type": slot['slot_type'],
                    "value": slot['value'],
                    "start": i,
                    "end": i + len(value_tokens)
                })
                break

    return {
        "command": filled_template,
        "intent": intent,
        "tokens": tokens,
        "bio_tags": ' '.join(bio_tags),
        "slots": final_slots,
        "token_count": len(tokens)
    }

def validate_entry(entry):
    """Validate BIO tag consistency."""
    tokens = entry['tokens']
    tags = entry['bio_tags'].split()

    if len(tokens) != len(tags):
        return False, f"Token/tag count mismatch: {len(tokens)} vs {len(tags)}"

    # Check B-I consistency
    prev_slot = None
    for tag in tags:
        if tag.startswith('I-'):
            slot = tag[2:]
            if prev_slot != slot:
                return False, f"I-{slot} without matching B-{slot}"
        if tag.startswith('B-') or tag.startswith('I-'):
            prev_slot = tag[2:]
        else:
            prev_slot = None

    return True, "OK"

def generate_dataset(target_total=5871, random_seed=42):
    """Generate the full ATIS-style dataset."""
    random.seed(random_seed)

    # Load taxonomy
    taxonomy = load_taxonomy()
    pools = taxonomy['slot_value_pools']
    intent_dist = taxonomy['intent_distribution']

    # Adjust distribution to match target total
    raw_total = sum(intent_dist.values())
    scale = target_total / raw_total
    intent_counts = {}
    running_total = 0
    intents_sorted = sorted(intent_dist.keys(), key=lambda k: intent_dist[k], reverse=True)

    for i, intent in enumerate(intents_sorted):
        if i == len(intents_sorted) - 1:
            intent_counts[intent] = target_total - running_total
        else:
            intent_counts[intent] = round(intent_dist[intent] * scale)
            running_total += intent_counts[intent]

    print(f"Target distribution (total={target_total}):")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count} ({count/target_total*100:.1f}%)")

    # Generate commands
    dataset = []
    all_commands = set()  # deduplicate

    for intent, target_count in tqdm(intent_counts.items(), desc="Generating intents"):
        templates = TEMPLATES.get(intent, [])
        if not templates:
            print(f"Warning: No templates for intent '{intent}'")
            continue

        generated = 0
        attempts = 0
        max_attempts = target_count * 10

        while generated < target_count and attempts < max_attempts:
            template = random.choice(templates)
            entry = generate_command_from_template(template, intent, pools)

            # Deduplicate
            if entry['command'] in all_commands:
                attempts += 1
                continue

            # Validate
            valid, msg = validate_entry(entry)
            if not valid:
                attempts += 1
                continue

            all_commands.add(entry['command'])
            dataset.append(entry)
            generated += 1
            attempts += 1

        if generated < target_count:
            print(f"Warning: Only generated {generated}/{target_count} for '{intent}' (not enough unique combinations)")

    random.shuffle(dataset)
    return dataset

def print_statistics(dataset):
    """Print dataset statistics matching original ATIS format."""
    print(f"\n{'='*60}")
    print(f"GENERATED DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total commands: {len(dataset)}")

    # Intent distribution
    intent_counts = Counter(e['intent'] for e in dataset)
    print(f"\nIntent distribution ({len(intent_counts)} intents):")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count} ({count/len(dataset)*100:.1f}%)")

    # Slot type coverage
    all_slots = set()
    slot_counts = Counter()
    for entry in dataset:
        for tag in entry['bio_tags'].split():
            if tag.startswith('B-'):
                slot = tag[2:]
                all_slots.add(slot)
                slot_counts[slot] += 1

    print(f"\nSlot types: {len(all_slots)}")
    print(f"\nSlot distribution (top 20):")
    for slot, count in slot_counts.most_common(20):
        print(f"  {slot}: {count}")

    # Token statistics
    token_counts = [e['token_count'] for e in dataset]
    print(f"\nToken statistics:")
    print(f"  Total tokens: {sum(token_counts)}")
    print(f"  Avg tokens/command: {sum(token_counts)/len(token_counts):.1f}")
    print(f"  Min: {min(token_counts)}, Max: {max(token_counts)}")

    # Entries with entities
    has_entity = sum(1 for e in dataset if 'B-' in e['bio_tags'])
    print(f"\nEntries with slots: {has_entity}/{len(dataset)} ({has_entity/len(dataset)*100:.1f}%)")

    # Validation
    errors = 0
    for e in dataset:
        valid, msg = validate_entry(e)
        if not valid:
            errors += 1
    print(f"Validation errors: {errors}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate ATIS-style source commands with full BIO annotations')
    parser.add_argument('--output', '-o', type=str,
                        default='data/multiatis_multilingual_pipeline/multiatis_commands_v3.json',
                        help='Output JSON file')
    parser.add_argument('--target_total', '-n', type=int, default=5871,
                        help='Target total commands (default: 5871 matching ATIS)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()

def main():
    args = parse_arguments()

    print(f"Generating {args.target_total} ATIS-style commands...")
    dataset = generate_dataset(target_total=args.target_total, random_seed=args.seed)

    # Print statistics
    print_statistics(dataset)

    # Save output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(dataset)} commands to {args.output}")

if __name__ == "__main__":
    main()
