"""
Process the skit-s2i synthetic dataset to create train/test splits in the format
expected by the baseline training scripts.

Output format (CSV):
- id: Unique identifier for each sample
- intent_class: Integer class label (0-13 for 14 intents)
- template: The text utterance
- audio_path: Path to the audio file
- speaker_id: Speaker identifier

This matches the format expected by speech-to-intent-dataset/baselines/dataset.py

Usage:
    python process_skit_s2i_dataset.py --metadata generated_audio/audio_metadata.json --output-dir output
"""

import json
import csv
import os
import argparse
import random
from collections import defaultdict


# Intent to class mapping (matching skit-s2i intent_info.csv)
INTENT_TO_CLASS = {
    "branch_address": 0,
    "activate_card": 1,
    "past_transactions": 2,
    "dispatch_status": 3,
    "outstanding_balance": 4,
    "card_issue": 5,
    "ifsc_code": 6,
    "generate_pin": 7,
    "unauthorised_transaction": 8,
    "loan_query": 9,
    "balance_enquiry": 10,
    "change_limit": 11,
    "block": 12,
    "lost": 13
}

# Intent descriptions for reference
INTENT_INFO = {
    0: {"intent": "branch_address", "description": "Bank branch location queries"},
    1: {"intent": "activate_card", "description": "Card activation requests"},
    2: {"intent": "past_transactions", "description": "Transaction history inquiries"},
    3: {"intent": "dispatch_status", "description": "Card/document dispatch status"},
    4: {"intent": "outstanding_balance", "description": "Outstanding dues queries"},
    5: {"intent": "card_issue", "description": "Card problem reports"},
    6: {"intent": "ifsc_code", "description": "IFSC code queries"},
    7: {"intent": "generate_pin", "description": "PIN generation requests"},
    8: {"intent": "unauthorised_transaction", "description": "Fraud reports"},
    9: {"intent": "loan_query", "description": "Loan-related queries"},
    10: {"intent": "balance_enquiry", "description": "Account balance checks"},
    11: {"intent": "change_limit", "description": "Transaction limit changes"},
    12: {"intent": "block", "description": "Card blocking requests"},
    13: {"intent": "lost", "description": "Lost card reports"}
}


def load_metadata(metadata_file):
    """Load audio metadata from JSON file."""
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_intent_info_csv(output_dir):
    """Create intent_info.csv matching the skit-s2i format."""
    output_file = os.path.join(output_dir, "intent_info.csv")
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["intent_class", "intent", "description"])
        writer.writeheader()
        
        for class_id, info in sorted(INTENT_INFO.items()):
            writer.writerow({
                "intent_class": class_id,
                "intent": info["intent"],
                "description": info["description"]
            })
    
    print(f"Created intent_info.csv: {output_file}")
    return output_file


def create_speaker_info_csv(output_dir, metadata):
    """Create speaker_info.csv mirroring the original skit-s2i/speaker_info.csv distribution.

    Real distribution (8F + 3M):
      Hindi x4, Bengali x3, Kannada x2, Malayalam x1, Punjabi x1
    """
    output_file = os.path.join(output_dir, "speaker_info.csv")

    speakers = set()
    for entry in metadata:
        speakers.add(entry.get("speaker_id", 0))

    # Aligned 1:1 with skit-s2i/speaker_info.csv
    speaker_info = {
        1:  {"gender": "Male",   "native_language": "Hindi",
             "languages_spoken": ["Hindi", "English"],
             "places_lived": ["Patna/Bihar"]},
        2:  {"gender": "Female", "native_language": "Bengali",
             "languages_spoken": ["English", "Hindi", "Bengali", "Odia"],
             "places_lived": ["Puri/Odisha"]},
        3:  {"gender": "Female", "native_language": "Kannada",
             "languages_spoken": ["English", "Kannada"],
             "places_lived": ["Davanagere/Karnataka"]},
        4:  {"gender": "Female", "native_language": "Hindi",
             "languages_spoken": ["English", "Odia"],
             "places_lived": ["Cuttack/Odisha", "Kolkata/West Bengal"]},
        5:  {"gender": "Female", "native_language": "Punjabi",
             "languages_spoken": ["English", "Hindi", "Punjabi"],
             "places_lived": ["Jalandhar/Punjab"]},
        6:  {"gender": "Female", "native_language": "Bengali",
             "languages_spoken": ["Bengali", "English", "Hindi"],
             "places_lived": ["Kolkata/West Bengal"]},
        7:  {"gender": "Female", "native_language": "Malayalam",
             "languages_spoken": ["English", "Malayalam"],
             "places_lived": ["Kollam/Kerala"]},
        8:  {"gender": "Male",   "native_language": "Kannada",
             "languages_spoken": ["Kannada", "English"],
             "places_lived": ["Mysore/Karnataka"]},
        9:  {"gender": "Female", "native_language": "Hindi",
             "languages_spoken": ["Hindi", "English", "Bengali", "Bihari"],
             "places_lived": ["Kolkata/West Bengal"]},
        10: {"gender": "Male",   "native_language": "Hindi",
             "languages_spoken": ["Hindi", "English"],
             "places_lived": ["Ranchi/Jharkhand"]},
        11: {"gender": "Female", "native_language": "Bengali",
             "languages_spoken": ["English", "Hindi", "Marathi"],
             "places_lived": ["Mumbai/Maharashtra"]},
    }

    fieldnames = ["speaker_id", "native_language", "languages_spoken", "places_lived", "gender"]
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid in sorted(speakers):
            info = speaker_info.get(sid, {"gender": "Unknown", "native_language": "Unknown",
                                          "languages_spoken": [], "places_lived": []})
            writer.writerow({
                "speaker_id": sid,
                "native_language": info["native_language"],
                "languages_spoken": json.dumps(info["languages_spoken"]),
                "places_lived": json.dumps(info["places_lived"]),
                "gender": info["gender"],
            })

    print(f"Created speaker_info.csv: {output_file}")
    return output_file


def split_dataset(metadata, train_ratio=0.885, seed=42):
    """
    Split dataset into train/test sets.
    
    skit-s2i has ~88.5% train (10,445) and ~11.5% test (1,400).
    We stratify by intent to ensure balanced splits.
    """
    random.seed(seed)
    
    # Group by intent
    by_intent = defaultdict(list)
    for entry in metadata:
        intent = entry.get("intent", "unknown")
        by_intent[intent].append(entry)
    
    train_data = []
    test_data = []
    
    # Split each intent proportionally
    for intent, entries in by_intent.items():
        random.shuffle(entries)
        split_idx = int(len(entries) * train_ratio)
        train_data.extend(entries[:split_idx])
        test_data.extend(entries[split_idx:])
    
    # Shuffle final datasets
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    return train_data, test_data


def create_dataset_csv(data, output_file, audio_dir):
    """Create dataset CSV in the format expected by baselines."""
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "intent_class", "template", "audio_path", "speaker_id"])
        writer.writeheader()
        
        for i, entry in enumerate(data):
            intent = entry.get("intent", "unknown")
            intent_class = INTENT_TO_CLASS.get(intent, -1)
            
            if intent_class == -1:
                print(f"Warning: Unknown intent '{intent}' for entry {i}")
                continue
            
            # New synthesize_speech.py writes "file"; older versions wrote "audio_file"
            audio_file = entry.get("audio_file") or entry.get("file") or ""
            audio_path = os.path.join(audio_dir, audio_file) if audio_dir else audio_file
            
            writer.writerow({
                "id": i,
                "intent_class": intent_class,
                # New synthesize_speech.py writes "command"; older versions wrote "text"/"original_command"
                "template": entry.get("text") or entry.get("command") or entry.get("original_command") or "",
                "audio_path": audio_path,
                "speaker_id": entry.get("speaker_id", 0)
            })
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Process skit-s2i synthetic dataset")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to audio_metadata.json from synthesize_speech.py")
    parser.add_argument("--audio-dir", type=str, default="generated_audio",
                        help="Path to audio files directory (for audio_path in CSV)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory for processed files")
    parser.add_argument("--train-ratio", type=float, default=0.885,
                        help="Train set ratio (default: 0.885 to match skit-s2i)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--relative-audio-path", action="store_true",
                        help="Use relative paths for audio files")
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file not found: {args.metadata}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metadata
    print(f"Loading metadata from: {args.metadata}")
    metadata = load_metadata(args.metadata)
    print(f"Loaded {len(metadata)} entries")
    
    # Create intent_info.csv
    create_intent_info_csv(args.output_dir)
    
    # Create speaker_info.csv
    create_speaker_info_csv(args.output_dir, metadata)
    
    # Split dataset
    print(f"\nSplitting dataset (train ratio: {args.train_ratio})...")
    train_data, test_data = split_dataset(metadata, args.train_ratio, args.seed)
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Determine audio path format
    audio_dir = args.audio_dir if not args.relative_audio_path else ""
    
    # Create train.csv
    train_file = os.path.join(args.output_dir, "train.csv")
    create_dataset_csv(train_data, train_file, audio_dir)
    print(f"\nCreated train.csv: {train_file}")
    
    # Create test.csv
    test_file = os.path.join(args.output_dir, "test.csv")
    create_dataset_csv(test_data, test_file, audio_dir)
    print(f"Created test.csv: {test_file}")
    
    # Print statistics
    print("\n")
    print("DATASET PROCESSING SUMMARY")
    
    print(f"\nSplit Statistics:")
    print(f"Total samples: {len(metadata)}")
    print(f"Train samples: {len(train_data)} ({len(train_data)/len(metadata)*100:.1f}%)")
    print(f"Test samples: {len(test_data)} ({len(test_data)/len(metadata)*100:.1f}%)")
    
    # Intent distribution in train set
    print(f"\nTrain Set Intent Distribution:")
    train_by_intent = defaultdict(int)
    for entry in train_data:
        train_by_intent[entry.get("intent", "unknown")] += 1
    for intent, count in sorted(train_by_intent.items()):
        class_id = INTENT_TO_CLASS.get(intent, -1)
        print(f"{intent} (class {class_id}): {count}")
    
    # Intent distribution in test set
    print(f"\nTest Set Intent Distribution:")
    test_by_intent = defaultdict(int)
    for entry in test_data:
        test_by_intent[entry.get("intent", "unknown")] += 1
    for intent, count in sorted(test_by_intent.items()):
        class_id = INTENT_TO_CLASS.get(intent, -1)
        print(f"{intent} (class {class_id}): {count}")
    
    # Speaker distribution
    print(f"\nSpeaker Distribution:")
    speaker_counts = defaultdict(int)
    for entry in metadata:
        speaker_counts[entry.get("speaker_id", 0)] += 1
    for sid, count in sorted(speaker_counts.items()):
        print(f"Speaker {sid}: {count}")
    
    print(f"\nOutput Files:")
    print(f"{os.path.join(args.output_dir, 'train.csv')}")
    print(f"{os.path.join(args.output_dir, 'test.csv')}")
    print(f"{os.path.join(args.output_dir, 'intent_info.csv')}")
    print(f"{os.path.join(args.output_dir, 'speaker_info.csv')}")
    
    print("\n")


if __name__ == "__main__":
    main()
