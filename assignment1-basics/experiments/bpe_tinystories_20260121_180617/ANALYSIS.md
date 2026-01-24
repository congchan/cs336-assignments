# BPE Tokenizer Training Analysis - TinyStories Dataset
## GPT-2 Byte-to-Unicode Encoding Explained
Think of it like building a house: Bytes are the raw bricks, Unicode is the blueprint that says which brick goes where, and Characters are the finished walls you actually see.

1. The Byte (The Raw Material)
A Byte is just a number between 0 and 255. On its own, a byte has no meaning. It’s just electricity stored in a specific pattern. To a computer, the number 65 could be a price, a coordinate, or a letter.

2. Unicode (The Master Map)
Because every country used to have its own way of assigning numbers to letters (which caused a lot of "gibberish" text), Unicode was created.

Unicode is a giant universal table that assigns a unique number (called a Code Point) to every character in every language, plus emojis and symbols.

The letter A is always Code Point 65.

The "Grinning Face" emoji 😄 is Code Point 128516.

3. Encoding (The Translation)
If Unicode has numbers like 128,516, we can't fit that into a single Byte (which only goes up to 255).

Encoding is the system used to turn those big Unicode numbers into a sequence of small bytes.

UTF-8 is the most popular encoding. It uses 1 byte for simple English letters (like A) but uses 3 or 4 bytes for complex things like emojis.

4. The "Character" (What you see)
A Character is the human-readable result. One character might be made of one byte (like a), or it might be made of four bytes (like 🍕).

### The Problem

When working with BPE tokenizers, we need to serialize byte sequences to text files (JSON, txt). But bytes 0-255 include problematic characters:

| Byte Range | Problem | Examples |
|------------|---------|----------|
| 0-31 | Control characters (unprintable) | Null (0), Tab (9), Newline (10) |
| 32 | Space (invisible, hard to debug) | ' ' |
| 127 | Delete character (unprintable) | DEL |

**Why this is bad:**
- Can't see what tokens contain (debugging nightmare)
- JSON doesn't handle some control characters well
- Space looks empty: `" the"` vs `"the"` are hard to distinguish visually

### The GPT-2 Solution: Shifting
In Python, chr(i) takes an integer and returns the Unicode character at that position. GPT-2's `bytes_to_unicode()` maps all 256 bytes to **visible, distinct Unicode characters**.

#### Strategy

1. **Keep 188 "nice" bytes as-is** (already printable):
   - `!` to `~` (33-126): Standard ASCII printable characters
   - `¡` to `¬` (161-172): Latin-1 supplement characters
   - `®` to `ÿ` (174-255): More Latin-1 characters

2. **Shift 68 "problematic" bytes by 256**:
   - These are control chars, whitespace, and a few others
   - Add 256 to get new Unicode code points (256-323)
   - These map to characters like: Ā, ā, Ă, ă, Ą, ą, Ć, ć, Ĉ, ĉ, Ċ, ċ, Č, č, ...

#### Why Shifting by 256 Works

**Unicode Structure:**
- Unicode has 1,114,112 total code points (U+0000 to U+10FFFF)
- First 256 code points (0-255) are ISO-8859-1 (Latin-1)
- Next range (256-383) contains Latin Extended-A characters

**The Math:**
```
Original byte range:     0 to 255
Shifted range:           256 to 323 (256 + 0 to 256 + 67)
```

**Why no collisions:**
- "Nice" bytes stay in range 0-255
- "Problematic" bytes shift to 256-323
- These ranges don't overlap!
- All resulting characters are printable and distinct

#### Concrete Examples

| Byte | Type | Original | Shifted To | Unicode Char | Readable? |
|------|------|----------|------------|--------------|-----------|
| 32 | Space | chr(32) = ' ' | chr(256) = 'Ġ' | Ġ | ✅ Visible |
| 97 | Letter 'a' | chr(97) = 'a' | (kept as-is) | a | ✅ Visible |
| 0 | Null | chr(0) = '\x00' | chr(256) = 'Ā' | Ā | ✅ Visible |
| 10 | Newline | chr(10) = '\n' | chr(266) = 'Ċ' | Ċ | ✅ Visible |
| 116 | Letter 't' | chr(116) = 't' | (kept as-is) | t | ✅ Visible |

#### Real Token Examples

**Token: " the" (space + "the")**
```
Bytes: [32, 116, 104, 101]
       ↓    ↓    ↓    ↓
       Ġ    t    h    e
Result: "Ġthe"
```

**Token: "hello"**
```
Bytes: [104, 101, 108, 108, 111]
       ↓    ↓    ↓    ↓    ↓
       h    e    l    l    o
Result: "hello"
```

**Token: "\n\n" (two newlines)**
```
Bytes: [10, 10]
       ↓   ↓
       Ċ   Ċ
Result: "ĊĊ"
```

### How to Use

#### Encoding (Bytes → String)

```python
def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))

# Usage
encoder = gpt2_bytes_to_unicode()
token_bytes = b' the'
token_str = ''.join(encoder[b] for b in token_bytes)
print(token_str)  # "Ġthe"
```

#### Decoding (String → Bytes)

```python
# Create reverse mapping
encoder = gpt2_bytes_to_unicode()
decoder = {v: k for k, v in encoder.items()}

# Usage
token_str = "Ġthe"
token_bytes = bytes([decoder[c] for c in token_str])
print(token_bytes)  # b' the'
```

### Benefits

1. **Human Readable**: Can inspect tokens in text editor
   - `"Ġthe"` is clearly "space + the"
   - `"hello"` is obviously "hello"

2. **JSON Safe**: All characters are valid in JSON strings
   - No escape sequences needed
   - No encoding issues

3. **Unambiguous**: Every byte has unique, visible representation
   - Space becomes `Ġ` (clearly visible)
   - Newline becomes `Ċ` (clearly visible)

4. **Standard**: Used by GPT-2, GPT-3, tiktoken, many others
   - Widely supported
   - Well-documented
   - Interoperable

### Vocabulary File Format

**Before (Raw Bytes - Not Readable):**
```json
{
  "256": [60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62],
  "257": [32, 116, 104, 101]
}
```

**After (GPT-2 Format - Readable):**
```json
{
  "<|endoftext|>": 256,
  "Ġthe": 257,
  "Ġand": 258,
  "hello": 259
}
```

Notice:
- Keys are now readable strings (not byte arrays)
- Values are token IDs (not byte arrays)
- Format is inverted: `{token: id}` instead of `{id: bytes}`
- Special tokens like `<|endoftext|>` are naturally represented

### Mathematical Proof of No Collisions

**Claim**: The mapping is bijective (one-to-one).

**Proof**:
1. Let A = set of 188 "nice" bytes that stay in range [0, 255]
2. Let B = set of 68 "problematic" bytes that shift to range [256, 323]
3. A ∩ B = ∅ (by construction, each byte is in exactly one set)
4. |A| + |B| = 188 + 68 = 256 (all bytes covered)
5. Range(A) = [0, 255] ∩ nice_chars
6. Range(B) = [256, 323]
7. Range(A) ∩ Range(B) = ∅ (no overlap)
8. Therefore, the mapping is injective (one-to-one)
9. Since |Domain| = |Range| = 256, it's also surjective
10. Thus, it's bijective ∎

### Why 256 Specifically?

The shift amount of 256 is chosen because:

1. **Power of 2**: 256 = 2^8, aligns with byte boundaries
2. **Beyond byte range**: Ensures no overlap with original 0-255 range
3. **Valid Unicode**: Code points 256-323 are all valid, printable Unicode characters (Latin Extended-A)
4. **Minimal shift**: Uses the smallest shift that guarantees no collisions

Could we use 512? Yes, but 256 is sufficient and more compact.

### Summary

GPT-2's byte-to-unicode encoding solves the problem of representing arbitrary bytes as human-readable text by:
- Keeping printable characters as-is (188 bytes)
- Shifting problematic characters by 256 (68 bytes)
- Ensuring all 256 bytes map to unique, visible, JSON-safe Unicode characters
- Providing a standard, reversible encoding used across the industry

This makes tokenizer vocabularies easy to inspect, debug, and share!


## Experiment Overview

**Dataset**: TinyStoriesV2-GPT4-train.txt (2.1 GB)
**Target Vocabulary Size**: 10,000
**Special Tokens**: `<|endoftext|>`
**Training Date**: January 21, 2026
**Experiment Directory**: `experiments/bpe_tinystories_20260121_180617/`

---

## Training Performance Metrics

### Time Performance

| Metric | Value |
|--------|-------|
| **Total Training Time** | **2.49 hours** (149.51 minutes / 8,970.60 seconds) |
| Number of Merges | 9,743 |
| Average Time per Merge | ~0.92 seconds |
| Final Vocabulary Size | 10,000 tokens |

**Answer to Question 2**: Training took approximately **2.5 hours** on the TinyStories dataset.

### Memory Usage

| Metric | Value |
|--------|-------|
| **Peak Memory Usage** | **87.80 MB** |
| Final Memory Usage | 4.62 MB |

**Answer to Question 2**: Peak memory usage was **87.80 MB** during training.

---

## Profiling Analysis - Performance Bottlenecks

### Answer to Question 3: What Takes the Most Time?

Based on the profiling data, the tokenizer training process has the following performance characteristics:

#### Top Time-Consuming Operations (by total time):

1. **`vanilla_merge()` function (bpe.py:83)** - **3,595.45 seconds (48.4% of execution time)**
   - Called 9,743 times (once per merge)
   - Average: 0.369 seconds per call
   - Cumulative: 7,036 seconds
   - **This is the PRIMARY bottleneck**

2. **List append operations** - **1,407.37 seconds (19.0%)**
   - Called 1,831,777,585 times
   - Used extensively in building new token sequences during merges

3. **`len()` built-in calls** - **1,273.69 seconds (17.2%)**
   - Called 4,247,365,529 times
   - Used for checking tuple/list lengths during merge operations

4. **`_get_stats()` function (bpe.py:62)** - **862.76 seconds (11.6%)**
   - Called 9,743 times (once per merge)
   - Average: 0.089 seconds per call
   - Computes byte-pair statistics for finding the most frequent pair

#### Performance Breakdown by Operation:

```
Total profiled time: 7,426 seconds (~2.06 hours)
├── vanilla_merge():        3,595s (48.4%) - Performing merge operations
├── list.append():          1,407s (19.0%) - Building new token sequences
├── len():                  1,273s (17.2%) - Length checks
├── _get_stats():             863s (11.6%) - Computing pair frequencies
├── threading/locks:          180s ( 2.4%) - Multiprocessing overhead
└── Other operations:         108s ( 1.4%) - Progress bars, I/O, etc.
```

### Key Insights:

1. **The merge operation itself (`vanilla_merge`) is the dominant bottleneck**, consuming nearly half of all execution time. This function:
   - Computes statistics on all byte pairs
   - Finds the most frequent pair
   - Creates new token sequences by replacing the pair

2. **List operations are expensive** - Nearly 20% of time is spent on list append operations, suggesting that:
   - Building new token sequences involves many list manipulations
   - Memory allocations for growing lists add overhead

3. **The algorithm is O(n) in vocabulary size** - Each merge requires:
   - Scanning all tokens in the frequency dictionary
   - Computing pair statistics
   - Rebuilding token sequences

4. **Multiprocessing helped initial data loading** - The chunk processing happened efficiently, but the iterative merge process cannot be easily parallelized

---

## Vocabulary Statistics

### File Sizes:
- **vocab.json**: 200 KB (10,000 tokens in GPT-2 format)
- **merges.txt**: 82 KB (9,743 merge operations in GPT-2 format)

**Note**: Files were converted to GPT-2 byte-to-unicode encoding format for human readability.

### Sample Merge Operations (First 10):

The most frequent byte pairs merged first (shown in GPT-2 format):

1. `Ġ t` → Space + 't' (common in English)
2. `h e` → 'h' + 'e' (forms "he")
3. `Ġ a` → Space + 'a' (common in English)
4. `Ġ s` → Space + 's' (common in English)
5. `Ġ w` → Space + 'w' (common in English)
6. `n d` → 'n' + 'd' (forms "nd")
7. `Ġt he` → "Ġt" + "he" (forms "Ġthe" = " the")
8. `e d` → 'e' + 'd' (forms "ed")
9. `Ġ b` → Space + 'b'
10. `Ġt o` → "Ġt" + 'o' (forms "Ġto" = " to")

**Format Note**: `Ġ` represents space (byte 32) in GPT-2 encoding. This makes tokens human-readable:
- `Ġthe` = " the" (space + "the")
- `hello` = "hello"

**Observation**: The tokenizer correctly learns common English patterns:
- Common words like "the", "was", "and"
- Common suffixes like "ed", "er"
- Frequent word beginnings with spaces

---

## Optimization Opportunities

Based on the profiling analysis, potential optimizations include:

1. **Optimize `vanilla_merge()` function**:
   - Use more efficient data structures (e.g., heaps for finding max)
   - Reduce redundant computations
   - Cache intermediate results

2. **Reduce list operations**:
   - Pre-allocate list sizes where possible
   - Use array-based structures instead of lists
   - Consider using NumPy arrays for better performance

3. **Optimize `_get_stats()` function**:
   - Use vectorized operations
   - Implement incremental statistics updates instead of full recomputation

4. **Consider algorithmic improvements**:
   - Implement priority queue for pair frequencies
   - Use incremental updates instead of full rescans

---

## Answers to Experimental Questions

### Q1: Serialize the resulting vocabulary and merges to disk
✅ **Completed**
- Vocabulary: `vocab.json` (200 KB, 10,000 tokens in GPT-2 format)
- Merges: `merges.txt` (82 KB, 9,743 merge operations in GPT-2 format)
- Format: GPT-2 byte-to-unicode encoding (human-readable)

### Q2: How many hours and memory did training take?
✅ **Completed**
- **Training Time**: 2.49 hours (149.51 minutes)
- **Peak Memory**: 87.80 MB
- **Average per Merge**: 0.92 seconds

### Q3: What part of the tokenizer training process takes the most time?
✅ **Completed**

**Primary Bottleneck**: The `vanilla_merge()` function (48.4% of time)
- This function performs the actual merge operations
- Called 9,743 times (once per vocabulary expansion)
- Takes ~0.37 seconds per call on average

**Secondary Bottlenecks**:
- List append operations (19.0%)
- Length checks via `len()` (17.2%)
- Statistics computation via `_get_stats()` (11.6%)

**Conclusion**: The iterative merge process dominates training time, with nearly 80% of time spent on merge operations and associated data structure manipulations.

---

## Files Generated

1. **training.log** - Complete timestamped log of training process
2. **vocab.json** - Final vocabulary (10,000 tokens in GPT-2 format)
3. **merges.txt** - All 9,743 merge operations in GPT-2 format
4. **profile_stats.txt** - Detailed profiling statistics
5. **training_summary.txt** - Quick summary of key metrics
6. **ANALYSIS.md** - This comprehensive analysis document
