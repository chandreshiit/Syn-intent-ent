"""
Check and validate audio files generated for the skit-s2i dataset.
This script verifies audio file integrity, statistics, and provides a summary.

Usage:
    python check_audio_files.py --audio-dir generated_audio --metadata audio_metadata.json
"""

import os
import json
import argparse
import wave
from pathlib import Path
from tqdm import tqdm
import statistics


def get_audio_info(file_path):
    """Get audio file information."""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / frame_rate
            
            return {
                "valid": True,
                "channels": n_channels,
                "sample_width": sample_width,
                "sample_rate": frame_rate,
                "n_frames": n_frames,
                "duration": duration,
                "file_size": os.path.getsize(file_path)
            }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }


def check_audio_files(audio_dir, metadata_file=None):
    """Check all audio files in the directory."""
    
    # Get all WAV files
    audio_files = list(Path(audio_dir).glob("*.wav"))
    
    if not audio_files:
        print(f"No WAV files found in {audio_dir}")
        return None
    
    print(f"Found {len(audio_files)} audio files")
    
    # Check each file
    valid_files = []
    invalid_files = []
    durations = []
    file_sizes = []
    sample_rates = set()
    
    for audio_file in tqdm(audio_files, desc="Checking audio files"):
        info = get_audio_info(str(audio_file))
        
        if info["valid"]:
            valid_files.append({
                "file": audio_file.name,
                **info
            })
            durations.append(info["duration"])
            file_sizes.append(info["file_size"])
            sample_rates.add(info["sample_rate"])
        else:
            invalid_files.append({
                "file": audio_file.name,
                **info
            })
    
    # Load and check metadata if provided
    metadata_info = None
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Check for missing audio files
        metadata_audio_files = {m["file"] for m in metadata}
        actual_audio_files = {f.name for f in audio_files}
        
        missing_files = metadata_audio_files - actual_audio_files
        extra_files = actual_audio_files - metadata_audio_files
        
        metadata_info = {
            "total_entries": len(metadata),
            "missing_files": list(missing_files)[:10],  # First 10
            "missing_count": len(missing_files),
            "extra_files": list(extra_files)[:10],  # First 10
            "extra_count": len(extra_files)
        }
        
        # Intent distribution
        intent_counts = {}
        for m in metadata:
            intent = m.get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        metadata_info["intent_distribution"] = intent_counts
        
        # Speaker distribution
        speaker_counts = {}
        for m in metadata:
            speaker = m.get("speaker_id", 0)
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        metadata_info["speaker_distribution"] = speaker_counts
    
    # Calculate statistics
    stats = {
        "total_files": len(audio_files),
        "valid_files": len(valid_files),
        "invalid_files": len(invalid_files),
        "sample_rates": list(sample_rates),
        "duration_stats": {
            "min": min(durations) if durations else 0,
            "max": max(durations) if durations else 0,
            "mean": statistics.mean(durations) if durations else 0,
            "median": statistics.median(durations) if durations else 0,
            "stdev": statistics.stdev(durations) if len(durations) > 1 else 0,
            "total_hours": sum(durations) / 3600 if durations else 0
        },
        "file_size_stats": {
            "min_kb": min(file_sizes) / 1024 if file_sizes else 0,
            "max_kb": max(file_sizes) / 1024 if file_sizes else 0,
            "mean_kb": statistics.mean(file_sizes) / 1024 if file_sizes else 0,
            "total_mb": sum(file_sizes) / (1024 * 1024) if file_sizes else 0
        },
        "metadata_info": metadata_info,
        "invalid_file_list": invalid_files[:20]  # First 20 invalid files
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Check audio files for skit-s2i dataset")
    parser.add_argument("--audio-dir", type=str, default="generated_audio",
                        help="Directory containing audio files")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Audio metadata JSON file (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for report (optional)")
    parser.add_argument("--fix", action="store_true",
                        help="Remove invalid audio files")
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.audio_dir):
        print(f"Error: Directory not found: {args.audio_dir}")
        return
    
    # If metadata not specified, look for default
    metadata_file = args.metadata
    if not metadata_file:
        default_metadata = os.path.join(args.audio_dir, "audio_metadata.json")
        if os.path.exists(default_metadata):
            metadata_file = default_metadata
    
    # Check audio files
    stats = check_audio_files(args.audio_dir, metadata_file)
    
    if not stats:
        return
    
    # Print report
    print("\n")
    print("AUDIO FILE CHECK REPORT")
    
    print(f"\nDirectory: {args.audio_dir}")
    print(f"\nFile Statistics:")
    print(f"Total files: {stats['total_files']}")
    print(f"Valid files: {stats['valid_files']}")
    print(f"Invalid files: {stats['invalid_files']}")
    
    print(f"\nAudio Properties:")
    print(f"Sample rates: {stats['sample_rates']}")
    
    ds = stats['duration_stats']
    print(f"\nDuration Statistics:")
    print(f"Min: {ds['min']:.2f}s")
    print(f"Max: {ds['max']:.2f}s")
    print(f"Mean: {ds['mean']:.2f}s")
    print(f"Median: {ds['median']:.2f}s")
    print(f"Std Dev: {ds['stdev']:.2f}s")
    print(f"Total: {ds['total_hours']:.2f} hours")
    
    fs = stats['file_size_stats']
    print(f"\nFile Size Statistics:")
    print(f"Min: {fs['min_kb']:.1f} KB")
    print(f"Max: {fs['max_kb']:.1f} KB")
    print(f"Mean: {fs['mean_kb']:.1f} KB")
    print(f"Total: {fs['total_mb']:.1f} MB")
    
    if stats['metadata_info']:
        mi = stats['metadata_info']
        print(f"\nMetadata Information:")
        print(f"Total entries: {mi['total_entries']}")
        print(f"Missing files: {mi['missing_count']}")
        print(f"Extra files: {mi['extra_count']}")
        
        if mi.get('intent_distribution'):
            print(f"\nIntent Distribution:")
            for intent, count in sorted(mi['intent_distribution'].items()):
                print(f"{intent}: {count}")
        
        if mi.get('speaker_distribution'):
            print(f"\nSpeaker Distribution:")
            for speaker, count in sorted(mi['speaker_distribution'].items()):
                print(f"Speaker {speaker}: {count}")
    
    if stats['invalid_files']:
        print(f"\nInvalid Files (first 20):")
        for f in stats['invalid_file_list']:
            print(f"{f['file']}: {f.get('error', 'Unknown error')}")
    
    # Fix invalid files if requested
    if args.fix and stats['invalid_files']:
        print(f"\nRemoving {stats['invalid_files']} invalid files...")
        for f in stats['invalid_file_list']:
            file_path = os.path.join(args.audio_dir, f['file'])
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed: {f['file']}")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
    print("\n")

if __name__ == "__main__":
    main()
