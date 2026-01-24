#!/usr/bin/env python3
"""
Train a byte-level BPE tokenizer on the TinyStories dataset.

This script:
1. Trains a BPE tokenizer with vocab size 10,000
2. Adds the <|endoftext|> special token
3. Serializes vocabulary and merges to disk
4. Records training time and memory usage
5. Profiles the code to identify bottlenecks
"""

import json
import time
import tracemalloc
import cProfile
import pstats
import logging
from pathlib import Path
from datetime import datetime
import io

from cs336_basics.bpe import train_bpe


def setup_logging(log_file):
    """
    Setup logging to both file and console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def gpt2_bytes_to_unicode():
    """
    Returns a mapping between every possible byte (0-255) to a printable unicode character.
    This is the GPT-2 encoding scheme for making bytes human-readable.

    WHY SHIFTING WORKS:

    The problem: Bytes 0-255 include control characters and whitespace that are:
      - Unprintable (can't see them in text files)
      - Problematic in JSON (newlines, tabs, null bytes)
      - Ambiguous (space looks empty, hard to debug)

    The solution: Map all 256 bytes to visible, distinct Unicode characters:
      1. Keep 188 "nice" printable characters as-is (!, ", #, ..., a, b, c, ...)
      2. Shift the remaining 68 "problematic" bytes by 256 to get new Unicode characters

    Why shifting by 256 works:
      - Unicode has 1,114,112 code points (U+0000 to U+10FFFF)
      - Bytes 0-255 occupy the first 256 code points
      - By adding 256, we move to code points 256-323 (Ā, ā, Ă, ă, ...)
      - These are all valid, printable, distinct Unicode characters
      - No collisions: original printable chars stay in 0-255, shifted chars in 256-323

    Example mappings:
      - Byte 32 (space ' ') -> chr(256 + 0) = chr(256) = 'Ā' (but actually 'Ġ' in GPT-2)
      - Byte 0 (null) -> chr(256) = 'Ā'
      - Byte 10 (newline) -> chr(256 + n) = some char in 256-323 range
      - Byte 97 ('a') -> chr(97) = 'a' (kept as-is, already printable)

    Result: Every byte has a unique, visible, JSON-safe representation!
    """
    # These 188 bytes are already printable and safe to use as-is
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]

    # For the remaining 68 bytes (control chars, whitespace), shift them by 256
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)  # Shift by 256 to get unique Unicode char
            n += 1

    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def serialize_vocab(vocab, output_path, logger):
    """
    Serialize vocabulary to JSON file using GPT-2 format.
    Format: {token_string: token_id}

    Uses GPT-2's byte-to-unicode mapping to make tokens human-readable.
    Example: bytes [32, 116, 104, 101] -> "Ġthe" (space + "the")
    """
    byte_encoder = gpt2_bytes_to_unicode()

    # Convert from {token_id: bytes} to {token_string: token_id}
    vocab_serializable = {}
    for token_id, token_bytes in vocab.items():
        # Convert bytes to string using GPT-2 encoding
        token_str = ''.join(byte_encoder[b] for b in token_bytes)
        vocab_serializable[token_str] = token_id

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, indent=4, ensure_ascii=False)

    logger.info(f"Vocabulary saved to {output_path} (GPT-2 format)")


def serialize_merges(merges, output_path, logger):
    """
    Serialize merges to text file using GPT-2 format.
    Format: Each line contains two space-separated tokens using GPT-2 byte encoding.

    Example: bytes [32, 116] + bytes [104, 101] -> "Ġt he"
    """
    byte_encoder = gpt2_bytes_to_unicode()

    with open(output_path, 'w', encoding='utf-8') as f:
        for token1_bytes, token2_bytes in merges:
            # Convert bytes to GPT-2 encoded strings
            token1_str = ''.join(byte_encoder[b] for b in token1_bytes)
            token2_str = ''.join(byte_encoder[b] for b in token2_bytes)
            f.write(f"{token1_str} {token2_str}\n")

    logger.info(f"Merges saved to {output_path} (GPT-2 format)")


def train_and_profile(input_path, vocab_size, special_tokens, experiment_name):
    """
    Train BPE tokenizer with profiling and timing.

    Args:
        input_path: Path to the training text file
        vocab_size: Maximum vocabulary size
        special_tokens: List of special tokens to add
        experiment_name: Name for this experiment (used in output directory)
    """
    # Create experiments directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments") / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "training.log"
    logger = setup_logging(log_file)

    logger.info("=" * 80)
    logger.info("BPE Tokenizer Training on TinyStories Dataset")
    logger.info("=" * 80)
    logger.info(f"Input file: {input_path}")
    logger.info(f"Target vocabulary size: {vocab_size}")
    logger.info(f"Special tokens: {special_tokens}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    logger.info("")

    # Start memory tracking
    tracemalloc.start()

    # Start timing
    start_time = time.time()

    # Create profiler
    profiler = cProfile.Profile()

    # Train with profiling
    logger.info("Starting BPE training with profiling...")
    profiler.enable()

    try:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
    finally:
        profiler.disable()

    # End timing
    end_time = time.time()

    # Get memory statistics
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate training time
    training_time_seconds = end_time - start_time
    training_time_hours = training_time_seconds / 3600
    training_time_minutes = training_time_seconds / 60

    logger.info("")
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Training time: {training_time_hours:.4f} hours ({training_time_minutes:.2f} minutes, {training_time_seconds:.2f} seconds)")
    logger.info(f"Current memory usage: {current_memory / 1024 / 1024:.2f} MB")
    logger.info(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB")
    logger.info(f"Final vocabulary size: {len(vocab)}")
    logger.info(f"Number of merges: {len(merges)}")
    logger.info("=" * 80)
    logger.info("")

    # Serialize results
    logger.info("Serializing vocabulary and merges...")
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"

    serialize_vocab(vocab, vocab_path, logger)
    serialize_merges(merges, merges_path, logger)

    # Save profiling results
    logger.info("")
    logger.info("=" * 80)
    logger.info("Profiling Results - Top 20 Time-Consuming Functions")
    logger.info("=" * 80)

    # Create string buffer for profiling output
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)

    profiling_output = s.getvalue()
    logger.info(profiling_output)

    # Save profiling results to file
    profile_path = output_dir / "profile_stats.txt"
    with open(profile_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BPE Tokenizer Training Profile\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training time: {training_time_hours:.4f} hours\n")
        f.write(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB\n\n")
        f.write("Top 20 Time-Consuming Functions (by cumulative time):\n")
        f.write("=" * 80 + "\n")
        f.write(profiling_output)
        f.write("\n\n")

        # Also print by total time
        s2 = io.StringIO()
        ps2 = pstats.Stats(profiler, stream=s2)
        ps2.strip_dirs()
        ps2.sort_stats('tottime')
        ps2.print_stats(20)
        f.write("Top 20 Time-Consuming Functions (by total time):\n")
        f.write("=" * 80 + "\n")
        f.write(s2.getvalue())

    logger.info(f"\nFull profiling results saved to {profile_path}")

    # Save summary
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BPE Tokenizer Training Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Target vocabulary size: {vocab_size}\n")
        f.write(f"Special tokens: {special_tokens}\n")
        f.write(f"Final vocabulary size: {len(vocab)}\n")
        f.write(f"Number of merges: {len(merges)}\n\n")
        f.write(f"Training time: {training_time_hours:.4f} hours\n")
        f.write(f"Training time: {training_time_minutes:.2f} minutes\n")
        f.write(f"Training time: {training_time_seconds:.2f} seconds\n\n")
        f.write(f"Current memory usage: {current_memory / 1024 / 1024:.2f} MB\n")
        f.write(f"Peak memory usage: {peak_memory / 1024 / 1024:.2f} MB\n")

    logger.info(f"Training summary saved to {summary_path}")

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"All results saved to: {output_dir}")
    logger.info("  - training.log: Complete training log")
    logger.info("  - vocab.json: Vocabulary mapping")
    logger.info("  - merges.txt: BPE merge operations")
    logger.info("  - profile_stats.txt: Detailed profiling information")
    logger.info("  - training_summary.txt: Training metrics summary")
    logger.info("=" * 80)

    return vocab, merges, output_dir


def train_bpe_tinystories(vocab_size=10000):
    """
    Train BPE tokenizer on TinyStories dataset.

    Args:
        vocab_size: Maximum vocabulary size (default: 10000)

    Returns:
        vocab: Dictionary mapping token IDs to bytes
        merges: List of merge operations
        output_dir: Path to experiment output directory
    """
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    experiment_name = "bpe_tinystories"

    return train_and_profile(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        experiment_name=experiment_name
    )


def train_bpe_owt(vocab_size=10000):
    """
    Train BPE tokenizer on OpenWebText (OWT) dataset.

    Args:
        vocab_size: Maximum vocabulary size (default: 10000)

    Returns:
        vocab: Dictionary mapping token IDs to bytes
        merges: List of merge operations
        output_dir: Path to experiment output directory
    """
    input_path = "data/owt_train.txt"
    special_tokens = ["<|endoftext|>"]
    experiment_name = "bpe_owt"

    return train_and_profile(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        experiment_name=experiment_name
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

        if dataset == "tinystories":
            train_bpe_tinystories(vocab_size=vocab_size)
        elif dataset == "owt":
            train_bpe_owt(vocab_size=vocab_size)
        else:
            print(f"Unknown dataset: {dataset}")
            print("Usage: python train_tinystories_bpe.py [tinystories|owt] [vocab_size]")
            sys.exit(1)
    else:
        # Default: train on TinyStories
        train_bpe_tinystories()
