import os
import regex as re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import BinaryIO
from tqdm import tqdm
import multiprocessing


BASE_VOCAB_SIZE = 256


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _get_stats(freqs):
    stats = defaultdict(int)
    for tuples in list(freqs):
        for pair in zip(tuples, tuples[1:]):
            # print(pair, freqs[tuples])
            stats[pair] += freqs[tuples]
    return stats


def init_vocab_data(base_vocab_size, special_tokens):
  chars2id = {chr(i): i for i in range(base_vocab_size)}
  id2bytes = {i: bytes([i]) for i in range(base_vocab_size)}
  bytes2id = {bytes([i]): i for i in range(base_vocab_size)}
  for special_token in special_tokens:
    special_token_id = len(chars2id)
    chars2id[special_token] = special_token_id
    id2bytes[special_token_id] = special_token.encode("utf-8")

  return chars2id, id2bytes, bytes2id


def vanilla_merge(freqs, id2bytes, bytes2id):
    stats = _get_stats(freqs)
    if not stats: # Check if stats is empty
        return freqs, None # Return current freqs and None for merged_pair

    (pair, freq) = max(stats.items(), key=lambda item: (item[1], item[0]))
    token1_bytes, token2_bytes = pair # e.g. b'h' (0x68) and b'e' (0x65)
    added_vocab_bytes = token1_bytes + token2_bytes  # e.g. b'he' (0x6865)
    idx = len(id2bytes)
    id2bytes[idx] = added_vocab_bytes
    bytes2id[added_vocab_bytes] = idx

    new_freqs = {}
    for tuples, freq in freqs.items():
        i = 0
        _new_key = []
        while i < len(tuples):
            if i < len(tuples) - 1 and (tuples[i], tuples[i + 1]) == pair:
                _new_key.append(added_vocab_bytes)
                i += 2
            else:
                _new_key.append(tuples[i])
                i += 1

        new_freqs[tuple(_new_key)] = freq

    return new_freqs, pair


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(input_path, special_tokens, start, end):
    """
    Runs in a separate process.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        sentences = [sentence for sentence in re.split(
              "|".join(token for token in special_tokens), chunk
            )
        ]
        local_frequency = defaultdict(int)
        for sentence in sentences:
          # Use re.finditer instead of re.findall
          for _match in re.finditer(PAT, sentence):
            _key = tuple(bytes([b]) for b in _match.group().encode("utf-8"))
            local_frequency[_key] += 1
    return local_frequency


def train_bpe(input_path, vocab_size, special_tokens, desired_num_chunks=20):
  """ Given a path to an input text file, trains a (byte-level) BPE tokenizer.
  BPE training function should handle (at least) the following input parameters:
    - input_path: str Path to a text file with BPE tokenizer training data.
    - vocab_size: int A positive integer that defines the maximum final vocabulary size (
      including the initial byte vocabulary, vocabulary items produced from merging,
      and any special tokens).
    - special_tokens: list[str] A list of strings to add to the vocabulary.
      These special tokens do not otherwise affect BPE training.

  Your BPE training function should return the resulting vocabulary and merges:
    - vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (
        token ID in the vocabulary) to bytes (token bytes).
    - merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training.
        Each list item is a tuple of bytes (<token1>, <token2>),
        representing that <token1> was merged with <token2>.
        The merges should be ordered by order of creation.

  """
  _, id2bytes, bytes2id = init_vocab_data(BASE_VOCAB_SIZE, special_tokens)
  special_token_bytes = special_tokens[0].encode("utf-8")
  with open(input_path, "rb") as f:
    boundaries = find_chunk_boundaries(f, desired_num_chunks, special_token_bytes)
    print(f"Got {len(boundaries)//2} chunks.")

  start_end_list = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
  # --- Prepare arguments for the pool ---
  # Each item is (input_path, start, end)
  args_list = [(input_path, special_tokens, start, end) for start, end in start_end_list]
  max_workers = multiprocessing.cpu_count()
  frequency = defaultdict(int)
  with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit the tasks to the process pool
    futures = {executor.submit(process_chunk, *args): args for args in args_list}
    # --- Gather results ---
    for future in as_completed(futures):
        local_frequency = future.result()
        for key, count in local_frequency.items():
            frequency[key] += count  # Fast and correct
  
  n_merges = vocab_size - len(id2bytes)
  print(f"Estimated {n_merges} merges.\nMerging...")
  merges = []
  for i in tqdm(range(n_merges)):
    frequency, merged_pair = vanilla_merge(frequency, id2bytes, bytes2id)
    if merged_pair is not None:
      token1_bytes, token2_bytes = merged_pair
      merges.append((token1_bytes, token2_bytes))
    else:
      print("No more pairs to merge. Maybe stopping early.")

  return id2bytes, merges
