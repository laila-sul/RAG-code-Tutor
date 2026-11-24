#!/usr/bin/env python3
"""
run_experiment.py  (v2)
==================================================================
Harness for the chapter "Engineering RAG-Driven Intelligent Tutoring
Systems: AST-aware chunking, prompt design and an efficiency index".

WHAT THIS SCRIPT IMPLEMENTS (stage -> code)
-------------------------------------------
Stage 1  Student profile + query ........ StudentProfile, QUERIES, classify_intent()
Stage 2  Structural retrieval ........... ast_semantic_chunker / fixed_size_chunker,
                                          BM25 + dense (384-d MiniLM) fused with RRF,
                                          cross-encoder re-ranking (Retriever)
Stage 3  Adaptive prompt injection ...... build_system_prompt() (profile + intent routing)
Stage 4  Inference backends ............. Gemini API, Groq, Cerebras (call_with_retries)
Stage 5  Evaluation ..................... simulated student (2-turn dialogue), automated
                                          tests, LLM judge -> P1, P2, P3, P
                                          (E and ITEI are computed in analyze_results.py)

DESIGN: 2 chunkers x 3 prompt strategies x 3 backends x Q queries x R repeats.
One CSV row per run, written incrementally (safe against crashes, --resume).

INTEGRITY FEATURES (so the paper cannot describe something that did not run)
---------------------------------------------------------------------------
* The CSV records the retriever mode that was ACTUALLY used (retriever_mode),
  the exact model IDs, the simulator/judge model IDs and all decoding settings.
* If the dense model or cross-encoder cannot be loaded, the script stops
  unless --allow-fallback is given; the fallback is then logged per row.
* --mock produces clearly-labelled fake data (status=MOCK) for testing the
  pipeline offline. analyze_results.py refuses MOCK data unless told otherwise.
* prompts_used.json and run_metadata.json are written next to the CSV so the
  prompts can go straight into an appendix.

SETUP
-----
    pip install openai google-genai pandas numpy
    pip install sentence-transformers          # dense retriever + cross-encoder
    export GEMINI_API_KEY=...  GROQ_API_KEY=...  CEREBRAS_API_KEY=...
    # model IDs can be overridden without editing the file:
    export GEMINI_MODEL=...  GROQ_MODEL=...  CEREBRAS_MODEL=...
    export SIMULATOR_PROVIDER=groq  SIMULATOR_MODEL=<an instruct model you can access>

USAGE
-----
    python run_experiment.py --selftest              # offline checks, no API calls
    python run_experiment.py --mock --quick          # pipeline smoke test (fake data)
    python run_experiment.py --quick                 # 1 query, 1 backend, real calls
    python run_experiment.py --repeats 3 --sleep 1   # full run
    python run_experiment.py --resume                # continue an interrupted run

SECURITY NOTE
-------------
Student-simulator code is executed in a separate Python process with a
timeout and memory/CPU limits. This is NOT a hardened sandbox; run the
harness in a container or VM if you do not trust the model output.
==================================================================
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import ast
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

# ------------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------------
try:
    import openai
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_OK = True
except ImportError:
    GENAI_OK = False
try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

# ==================================================================
# 0. CONFIGURATION (edit here or override with environment variables)
# ==================================================================
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

DEFAULT_MODELS = {"gemini": GEMINI_MODEL, "groq": GROQ_MODEL, "cerebras": CEREBRAS_MODEL}
KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY"}

# Student simulator / judge: set to the model you can actually access.
# (The chapter names Llama-3.1-70B-Instruct; if that endpoint is no longer
#  available, state the model you really used.)
SIMULATOR_PROVIDER = os.environ.get("SIMULATOR_PROVIDER", "groq")
SIMULATOR_MODEL = os.environ.get("SIMULATOR_MODEL", GROQ_MODEL)

TUTOR_TEMPERATURE = 0.0
SIMULATOR_TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "6000"))  # includes hidden reasoning tokens
MAX_RETRIES = 3
GEMINI_TIMEOUT_S = float(os.environ.get("GEMINI_TIMEOUT_S", "90"))  # per-request timeout

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"          # 384-dimensional
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ==================================================================
# 1. BENCHMARK CODEBASE
# ==================================================================
# NOTE: the bug-describing docstrings of v1 were removed. In v1 the retrieved
# context itself told the model what the bug was ("range stops one element
# early"), which leaks the answer to every tutor. The bug explanations now live
# only in the teacher-side `gold_notes` of each query.
SAMPLE_CODEBASE = '''
def factorial(n):
    """Recursive factorial. Base case n <= 1."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n):
    """Naive recursive Fibonacci (exponential branching)."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def merge_sort(arr):
    """Classic recursive divide-and-conquer merge sort."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left, right):
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def binary_search(arr, target):
    """Iterative binary search over a sorted list. Returns index or -1."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def linear_search_buggy(arr, target):
    """Linear search over a list. Returns the index of target or -1."""
    for i in range(len(arr) - 1):
        if arr[i] == target:
            return i
    return -1


def bubble_sort_buggy(arr):
    """Bubble sort. Sorts the list in place and returns it."""
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - i - 2):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


class Stack:
    """A minimal LIFO stack used for conceptual curriculum questions."""

    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        return self._items[-1] if self._items else None

    def is_empty(self):
        return len(self._items) == 0
'''.strip("\n")

# Reference fixes (used only by --selftest and by the unit tests)
REFERENCE_FIXES = {
    "linear_search_buggy": '''def linear_search_buggy(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1''',
    "bubble_sort_buggy": '''def bubble_sort_buggy(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr''',
}

TESTS = {
    "linear_search_buggy": (
        "assert linear_search_buggy([1, 2, 3], 3) == 2\n"
        "assert linear_search_buggy([1, 2, 3], 1) == 0\n"
        "assert linear_search_buggy([5], 5) == 0\n"
        "assert linear_search_buggy([], 1) == -1\n"
        "assert linear_search_buggy([1, 2, 3], 9) == -1\n"
    ),
    "bubble_sort_buggy": (
        "assert bubble_sort_buggy([3, 2, 1]) == [1, 2, 3]\n"
        "assert bubble_sort_buggy([5, 1, 4, 2, 8]) == [1, 2, 4, 5, 8]\n"
        "assert bubble_sort_buggy([2, 1]) == [1, 2]\n"
        "assert bubble_sort_buggy([1]) == [1]\n"
        "assert bubble_sort_buggy([]) == []\n"
    ),
}


# ==================================================================
# 2. CHUNKERS (both record character offsets for retrieval metrics)
# ==================================================================
def _line_offsets(source: str):
    offs, pos = [], 0
    for line in source.splitlines(keepends=True):
        offs.append(pos)
        pos += len(line)
    return offs


class CodeChunkVisitor(ast.NodeVisitor):
    """One chunk per top-level function/class (does not recurse into bodies)."""

    def __init__(self, source: str):
        self.source = source
        self.offs = _line_offsets(source)
        self.chunks = []

    def _add(self, node, node_type):
        seg = ast.get_source_segment(self.source, node) or ""
        start = self.offs[node.lineno - 1] + node.col_offset
        end = start + len(seg)
        self.chunks.append({
            "chunk_id": f"{node_type}:{node.name}:L{node.lineno}-{node.end_lineno}",
            "name": node.name, "node_type": node_type,
            "start_char": start, "end_char": end, "content": seg,
        })

    def visit_FunctionDef(self, node):
        self._add(node, "FunctionDef")

    def visit_AsyncFunctionDef(self, node):
        self._add(node, "AsyncFunctionDef")

    def visit_ClassDef(self, node):
        self._add(node, "ClassDef")


def ast_semantic_chunker(source: str, chunk_size=500, overlap=50):
    """AST chunking. Falls back to fixed windows when the code does not parse
    (typical for broken student code); the fallback is flagged in the chunk."""
    try:
        v = CodeChunkVisitor(source)
        v.visit(ast.parse(source))
        return v.chunks
    except SyntaxError:
        chunks = fixed_size_chunker(source, chunk_size, overlap)
        for c in chunks:
            c["node_type"] = "FallbackWindow"
        return chunks


def fixed_size_chunker(source: str, chunk_size=500, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks, start, idx, step = [], 0, 0, chunk_size - overlap
    while start < len(source):
        end = min(start + chunk_size, len(source))
        chunks.append({
            "chunk_id": f"fixed:{idx}:C{start}-{end}", "name": f"fixed_{idx}",
            "node_type": "FixedWindow", "start_char": start, "end_char": end,
            "content": source[start:end],
        })
        idx += 1
        if end == len(source):
            break
        start += step
    return chunks


def function_spans(source: str):
    """name -> (start_char, end_char) for every top-level function/class."""
    return {c["name"]: (c["start_char"], c["end_char"])
            for c in ast_semantic_chunker(source)}


# ==================================================================
# 3. RETRIEVAL: BM25 + dense (RRF fusion) + cross-encoder re-ranking
# ==================================================================
STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being
i me my we our you your it its this that these those do does did can could would should will
what whats how why when which who whom there here so than then also just about into out up down over
get got have has had not no yes very really actually sometimes occasionally""".split())


def tokenize(text: str):
    toks = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        for part in w.split("_"):
            for sub in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", part):
                s = sub.lower()
                if s and s not in STOPWORDS and len(s) > 1:
                    toks.append(s)
    return toks


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        df = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query):
        q = tokenize(query)
        out = []
        for d in self.docs:
            s, dl = 0.0, len(d)
            for t in q:
                f = d.count(t)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl)))
            out.append(s)
        return out


_ST_CACHE = {}


def _load_dense():
    if "dense" not in _ST_CACHE:
        from sentence_transformers import SentenceTransformer
        _ST_CACHE["dense"] = SentenceTransformer(EMBED_MODEL)
    return _ST_CACHE["dense"]


def _load_cross():
    if "cross" not in _ST_CACHE:
        from sentence_transformers import CrossEncoder
        _ST_CACHE["cross"] = CrossEncoder(RERANK_MODEL)
    return _ST_CACHE["cross"]


def _rank_of(scores):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    r = [0] * len(scores)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


class Retriever:
    """mode='hybrid' : BM25 + dense (RRF) candidates, cross-encoder re-rank
       mode='bm25'   : BM25 only
       The same retriever is used for BOTH chunkers so only the chunk
       boundaries differ between experimental arms."""

    def __init__(self, chunks, mode="hybrid", allow_fallback=False, candidate_k=5):
        self.chunks = chunks
        self.candidate_k = candidate_k
        self.bm25 = BM25([c["content"] for c in chunks])
        self.dense = self.cross = self.emb = None
        self.mode_actual = "bm25"
        if mode == "hybrid":
            try:
                self.dense = _load_dense()
                self.cross = _load_cross()
                self.emb = self.dense.encode([c["content"] for c in chunks],
                                             normalize_embeddings=True)
                self.mode_actual = "bm25+dense+crossencoder"
            except Exception as exc:  # noqa: BLE001
                if not allow_fallback:
                    raise RuntimeError(
                        "Hybrid retriever unavailable (%s: %s). Install sentence-transformers "
                        "and make sure the models can be downloaded, or pass --allow-fallback "
                        "(the fallback is logged and must be disclosed in the paper)."
                        % (type(exc).__name__, exc))
                self.dense = self.cross = self.emb = None
                self.mode_actual = "bm25(fallback)"

    def retrieve(self, query, top_k=2):
        n = len(self.chunks)
        b = self.bm25.scores(query)
        if self.mode_actual != "bm25+dense+crossencoder":
            order = sorted(range(n), key=lambda i: -b[i])[:top_k]
            return [self.chunks[i] for i in order if self.chunks[i]["content"].strip()]
        qe = self.dense.encode([query], normalize_embeddings=True)[0]
        d = [float(sum(x * y for x, y in zip(qe, e))) for e in self.emb]
        rb, rd = _rank_of(b), _rank_of(d)
        rrf = [1 / (60 + rb[i]) + 1 / (60 + rd[i]) for i in range(n)]
        cand = sorted(range(n), key=lambda i: -rrf[i])[:min(self.candidate_k, n)]
        ce = self.cross.predict([(query, self.chunks[i]["content"]) for i in cand])
        order = [i for _, i in sorted(zip(ce, cand), key=lambda t: -t[0])][:top_k]
        return [self.chunks[i] for i in order if self.chunks[i]["content"].strip()]


def build_context(chunks):
    return "\n\n# --- retrieved chunk boundary ---\n\n".join(c["content"] for c in chunks)


def retrieval_metrics(retrieved, source, target_names):
    """Does the retrieved context contain the function(s) the question is about?
       coverage = fraction of the target characters covered by retrieved chunks
       intact   = every target function lies completely inside ONE chunk"""
    spans = function_spans(source)
    cov_num = cov_den = 0
    intact_all = True
    for name in target_names:
        s, e = spans[name]
        cov_den += e - s
        covered = set()
        for c in retrieved:
            lo, hi = max(s, c["start_char"]), min(e, c["end_char"])
            covered.update(range(lo, hi)) if hi > lo else None
        cov_num += len(covered)
        intact_all &= any(c["start_char"] <= s and c["end_char"] >= e for c in retrieved)
    return {"target_coverage": round(cov_num / max(1, cov_den), 4),
            "target_intact": int(intact_all)}


# ==================================================================
# 4. TEST SUITE (queries + gold information for automatic evaluation)
# ==================================================================
@dataclass
class StudentProfile:
    prior_errors: list = field(default_factory=list)
    cognitive_load: str = "unknown"

    def updated(self, success: bool, error_tag: str | None):
        """Profile after the dialogue: adds the error tag when the student failed."""
        pe = list(self.prior_errors)
        if not success and error_tag and error_tag not in pe:
            pe.append(error_tag)
        return StudentProfile(pe, self.cognitive_load)


CODEBLOCK_RE = re.compile(r"```")

QUERIES = [
    {
        "query_id": "Q1_callstack_factorial", "category": "call_stack_tracing",
        "intent_gold": "tracing", "targets": ["factorial"],
        "query_text": ("Can you trace the call stack for factorial(5)? I understand the recursive "
                       "call going down, but I get confused about how the multiplication actually "
                       "happens as the calls return back up."),
        "profile": {"prior_errors": ["stack_overflow_misunderstanding"], "cognitive_load": "low"},
        "gold_notes": ("factorial(5) recurses down to factorial(1)=1. On the way back up the returns are "
                       "factorial(2)=2, factorial(3)=6, factorial(4)=24, factorial(5)=120. The "
                       "multiplication happens after each inner call returns, not on the way down."),
        "leak_patterns": [r"\b120\b"], "debug": False,
        "student_task": ("In your own words, write the value returned by each call on the way back up "
                         "and the final result of factorial(5)."),
        "student_check": [r"\b120\b", r"\b24\b", r"\b6\b"],
    },
    {
        "query_id": "Q2_callstack_mergesort", "category": "call_stack_tracing",
        "intent_gold": "tracing", "targets": ["merge_sort", "_merge"],
        "query_text": ("I'm tracing merge_sort([5, 3, 8, 1]) by hand and I lose track of which sub-list "
                       "is 'left' and which is 'right' once the recursion goes more than two levels "
                       "deep. Can you help me trace it?"),
        "profile": {"prior_errors": ["recursion_base_case_confusion"], "cognitive_load": "high"},
        "gold_notes": ("[5,3,8,1] splits into left [5,3] and right [8,1]; these split into [5],[3] and "
                       "[8],[1]. Merging gives [3,5] and [1,8], then the final [1,3,5,8]."),
        "leak_patterns": [r"\[\s*1\s*,\s*3\s*,\s*5\s*,\s*8\s*\]"], "debug": False,
        "student_task": ("Write the left and right sub-lists at each recursion level and the final "
                         "merged list."),
        "student_check": [r"\[\s*5\s*,\s*3\s*\]", r"\[\s*8\s*,\s*1\s*\]",
                          r"\[\s*1\s*,\s*3\s*,\s*5\s*,\s*8\s*\]"],
    },
    {
        "query_id": "Q3_debug_offbyone_linearsearch", "category": "off_by_one_debugging",
        "intent_gold": "debugging", "targets": ["linear_search_buggy"],
        "query_text": ("My linear_search_buggy function never finds the target when it's the very last "
                       "element in the list. Why does that happen?"),
        "profile": {"prior_errors": ["off_by_one_range"], "cognitive_load": "medium"},
        "gold_notes": ("Bug: range(len(arr) - 1) stops one element early, so index len(arr)-1 is never "
                       "checked. Fix: range(len(arr))."),
        "leak_patterns": [r"range\(\s*len\(\s*arr\s*\)\s*\)",
                          r"(remove|drop|delete|get rid of)\s+(the\s+)?[`'\"]?\s*-\s*1"],
        "debug": True, "tests_for": "linear_search_buggy",
        "student_task": ("Reply with your best attempt at a corrected version of linear_search_buggy: the "
                         "COMPLETE function in one ```python block with the same name, then one sentence "
                         "on what you changed."),
    },
    {
        "query_id": "Q4_debug_loopinvariant_bubblesort", "category": "loop_invariant_debugging",
        "intent_gold": "debugging", "targets": ["bubble_sort_buggy"],
        "query_text": ("Sometimes bubble_sort_buggy doesn't fully sort the array -- the last two "
                       "elements are occasionally still out of order. What's going wrong with my loop "
                       "bounds?"),
        "profile": {"prior_errors": ["loop_invariant_misunderstanding"], "cognitive_load": "medium"},
        "gold_notes": ("Bug: the inner loop bound n - i - 2 is one too small; it should be n - i - 1 so "
                       "that every adjacent pair up to the unsorted boundary is compared."),
        "leak_patterns": [r"n\s*-\s*i\s*-\s*1"],
        "debug": True, "tests_for": "bubble_sort_buggy",
        "student_task": ("Reply with your best attempt at a corrected version of bubble_sort_buggy: the "
                         "COMPLETE function in one ```python block with the same name, then one sentence "
                         "on what you changed."),
    },
    {
        "query_id": "Q5_concept_search_complexity", "category": "curriculum_alignment",
        "intent_gold": "concept", "targets": ["binary_search"],
        "query_text": ("What's the practical difference between binary_search and a plain linear search "
                       "in terms of time complexity, and when would I actually choose one over the other?"),
        "profile": {"prior_errors": [], "cognitive_load": "low"},
        "gold_notes": ("Binary search is O(log n) but needs a sorted list; linear search is O(n) and "
                       "works on unsorted data. Choose linear for small or unsorted data."),
        "leak_patterns": [r"O\(\s*log"], "debug": False,
        "student_task": ("Summarise in two sentences the time complexity of each and when you would "
                         "choose each."),
        "student_check": [r"log", r"O\(\s*n\s*\)|linear", r"sorted"],
    },
    {
        "query_id": "Q6_concept_stack_lifo", "category": "curriculum_alignment",
        "intent_gold": "concept", "targets": ["Stack"],
        "query_text": ("Can you explain how the Stack class enforces LIFO ordering through its push and "
                       "pop methods, conceptually?"),
        "profile": {"prior_errors": [], "cognitive_load": "low"},
        "gold_notes": ("push appends to the end of the list; pop removes and returns the last element, "
                       "so the most recently pushed item is the first one popped."),
        "leak_patterns": [r"removes?\s+(and returns\s+)?the\s+(last|most recent)",
                          r"(last|most recent)\w*\s+(item|element)?\s*(pushed|added|appended)"],
        "debug": False,
        "student_task": ("In your own words, say which item pop() returns after push(1), push(2), "
                         "push(3), and why."),
        "student_check": [r"\b3\b", r"last|most recent|top|end"],
    },
]

INTENT_MAP = {"call_stack_tracing": "tracing", "off_by_one_debugging": "debugging",
              "loop_invariant_debugging": "debugging", "curriculum_alignment": "concept"}


# ==================================================================
# 5. STAGE 1/3: INTENT ROUTING + ADAPTIVE PROMPTS
# ==================================================================
INTENT_RULES = [
    ("debugging", re.compile(r"\b(bug|buggy|never finds|doesn'?t (fully )?(sort|work)|going wrong|"
                             r"out of order|wrong|fix)\b", re.I)),
    ("tracing", re.compile(r"\b(trace|tracing|call stack|recurs\w*)\b", re.I)),
]


def classify_intent(text: str) -> str:
    """Rule-based intent router (debugging / tracing / concept)."""
    for label, rx in INTENT_RULES:
        if rx.search(text):
            return label
    return "concept"


INTENT_LINES = {
    "debugging": "The student suspects a bug. Help them locate it by examining concrete values.",
    "tracing": "The student wants to follow execution state. Focus on call/return order and values.",
    "concept": "The student asks a conceptual question. Connect it to the retrieved code.",
}


def strategy_zero_shot(profile, intent, variant="neutral"):
    if variant == "direct":  # the v1 baseline (instructs the model to give corrected code)
        return ("You are an AI coding tutor assistant embedded in an intelligent tutoring system. "
                "Answer the student's question directly and completely. Provide corrected code and "
                "explicit fixes where relevant. Be accurate and concise.")
    return ("You are an AI coding tutor. Answer the student's question using the retrieved code "
            "context.")


def strategy_socratic(profile: StudentProfile, intent):
    return (
        "You are an AI coding tutor practicing strict Socratic scaffolding.\n\n"
        "HARD CONSTRAINTS:\n"
        "1. NEVER provide solution code, a corrected snippet, or the exact fixed syntax.\n"
        "2. NEVER state the bug's location and fix in one sentence.\n"
        "3. Respond ONLY with 2-4 targeted guiding questions that lead the student to discover the "
        "issue themselves (e.g. ask them to trace a specific loop index or recursive call).\n"
        "4. You may reference line numbers or variable names, but never supply the corrected value "
        "or code.\n\n"
        f"{INTENT_LINES[intent]}\n"
        f"Known student profile: prior_errors={profile.prior_errors}, "
        f"cognitive_load={profile.cognitive_load}.\n"
        "If cognitive_load is 'high', ask fewer, simpler questions and add one sentence of "
        "encouragement. If the student has a matching prior_error, gently reference that pattern "
        "without naming the fix.")


def strategy_cot(profile: StudentProfile, intent):
    return (
        "You are an AI coding tutor. Before writing anything for the student, perform a rigorous "
        "step-by-step trace of the relevant variable state (loop indices, recursive call arguments "
        "and return values, or array contents at each step).\n\n"
        "Structure your ENTIRE response with exactly these two tags, in this order:\n\n"
        "<reasoning>\n  ... the detailed step-by-step state trace (internal verification, not shown "
        "to the student) ...\n</reasoning>\n"
        "<hint>\n  ... a concise, scaffolded hint grounded strictly in the trace. Do not give away "
        "more than needed for the student's level ...\n</hint>\n\n"
        f"{INTENT_LINES[intent]}\n"
        f"Known student profile: prior_errors={profile.prior_errors}, "
        f"cognitive_load={profile.cognitive_load}. Calibrate the <hint> to this profile.")


PROMPT_NAMES = ["zero_shot", "socratic", "cot_tracing"]


def build_system_prompt(name, profile, intent, zs_variant="neutral"):
    if name == "zero_shot":
        return strategy_zero_shot(profile, intent, zs_variant)
    if name == "socratic":
        return strategy_socratic(profile, intent)
    if name == "cot_tracing":
        return strategy_cot(profile, intent)
    raise ValueError(name)


def build_user_prompt(query_text, context):
    return ("### Retrieved Codebase Context\n"
            f"{context if context.strip() else '(no context retrieved)'}\n\n"
            "### Student Question\n"
            f"{query_text}\n")


def split_tutor_output(prompt_name, text):
    """Return (student_visible, hidden_reasoning, format_ok)."""
    if prompt_name != "cot_tracing":
        return text, "", True
    hint = re.search(r"<hint>(.*?)</hint>", text, re.S | re.I)
    reas = re.search(r"<reasoning>(.*?)</reasoning>", text, re.S | re.I)
    if hint:
        return hint.group(1).strip(), (reas.group(1).strip() if reas else ""), True
    stripped = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.S | re.I).strip()
    return stripped, (reas.group(1).strip() if reas else ""), False


# ==================================================================
# 6. STAGE 4: BACKENDS (uniform interface, retries logged, tokens normalised)
# ==================================================================
def _empty(status, msg=""):
    return {"text": msg, "input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
            "total_tokens": None, "latency_seconds": None, "status": status, "n_retries": 0,
            "truncated": None}


class GeminiBackend:
    provider = "gemini"

    def __init__(self, model):
        self.model = model
        self.client = None

    def chat(self, system, user, temperature, role="tutor"):
        if not GENAI_OK:
            return _empty("SDK_MISSING", "google-genai not installed")
        key = os.environ.get(KEY_ENV["gemini"])
        if not key:
            return _empty("API_KEY_MISSING", "GEMINI_API_KEY not set")
        if self.client is None:
            # Fail fast: hard per-request timeout and NO hidden SDK retries, so a throttled
            # call is recorded as an error instead of silently taking minutes.
            opts = {"timeout": int(GEMINI_TIMEOUT_S * 1000)}
            retry_cls = getattr(genai_types, "HttpRetryOptions", None)
            if retry_cls is not None:
                opts["retry_options"] = retry_cls(attempts=1)
            self.client = genai.Client(api_key=key, http_options=genai_types.HttpOptions(**opts))
        t0 = time.perf_counter()
        r = self.client.models.generate_content(
            model=self.model, contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system, temperature=temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True)))
        lat = time.perf_counter() - t0
        u = getattr(r, "usage_metadata", None)
        pin = getattr(u, "prompt_token_count", None) or 0
        out = getattr(u, "candidates_token_count", None) or 0
        thought = getattr(u, "thoughts_token_count", None) or 0
        total = getattr(u, "total_token_count", None) or (pin + out + thought)
        try:
            fr = str(r.candidates[0].finish_reason)
        except Exception:  # noqa: BLE001
            fr = ""
        truncated = "MAX_TOKENS" in fr.upper()
        return {"truncated": truncated, "text": r.text or "", "input_tokens": pin, "output_tokens": out,
                "reasoning_tokens": thought, "total_tokens": total,
                "latency_seconds": round(lat, 4), "status": "OK", "n_retries": 0}


class OpenAICompatBackend:
    def __init__(self, provider, model, base_url):
        self.provider, self.model, self.base_url, self.client = provider, model, base_url, None

    def chat(self, system, user, temperature, role="tutor"):
        if not OPENAI_OK:
            return _empty("SDK_MISSING", "openai not installed")
        key = os.environ.get(KEY_ENV[self.provider])
        if not key:
            return _empty("API_KEY_MISSING", f"{KEY_ENV[self.provider]} not set")
        if self.client is None:
            # max_retries=0: retries are handled (and logged) by call_with_retries,
            # so hidden SDK retries cannot inflate the measured latency.
            self.client = openai.OpenAI(api_key=key, base_url=self.base_url, max_retries=0)
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        lat = time.perf_counter() - t0
        u = getattr(r, "usage", None)
        details = getattr(u, "completion_tokens_details", None)
        return {"truncated": getattr(r.choices[0], "finish_reason", None) == "length",
                "text": r.choices[0].message.content or "",
                "input_tokens": getattr(u, "prompt_tokens", None),
                "output_tokens": getattr(u, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
                "total_tokens": getattr(u, "total_tokens", None),
                "latency_seconds": round(lat, 4), "status": "OK", "n_retries": 0}


class MockBackend:
    """Deterministic FAKE backend for offline pipeline tests. Output is labelled MOCK."""
    def __init__(self, provider, model):
        self.provider, self.model = provider, model

    def _h(self, *parts):
        return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest(), 16)

    def chat(self, system, user, temperature, role="tutor"):
        h = self._h(system, user, self.provider, role)
        if role == "judge":
            text = json.dumps({"accurate": h % 5 != 0, "gave_full_answer": "NEVER provide" not in
                               user and h % 3 == 0, "reason": "mock"})
        elif role == "student":
            fn = ("linear_search_buggy" if "linear_search_buggy" in user and "COMPLETE" in user else
                  "bubble_sort_buggy" if "bubble_sort_buggy" in user and "COMPLETE" in user else None)
            if fn:
                good = (h % 2 == 0)
                text = "```python\n" + (REFERENCE_FIXES[fn] if good else
                                        SAMPLE_CODEBASE.split("def " + fn)[1].split("\n\n\n")[0]
                                        .join(["def " + fn, ""])) + "\n```\nChanged the loop bound."
            else:
                text = "factorial(5)=120 with 24 and 6; [5, 3] [8, 1] [1, 3, 5, 8]; O(log n) vs O(n) sorted; 3 last"
        else:
            if "Socratic" in system:
                text = "What is the value of i on the last iteration? What index is never visited?"
            elif "<reasoning>" in system:
                text = "<reasoning>step 1 ... step 2 ...</reasoning><hint>Check the final index.</hint>"
            else:
                text = "The answer is 120. Use ```python\nrange(len(arr))\n``` to fix it."
        scale = {"gemini": 2.0, "groq": 0.6, "cerebras": 0.4}.get(self.provider, 1.0)
        toks = {"socratic": 700, "reasoning": 1350}.get(
            "reasoning" if "<reasoning>" in system else "socratic" if "Socratic" in system else "", 500)
        lat = round(scale * (0.3 + (toks / 1000) * 0.6) * (0.9 + (h % 20) / 100), 4)
        return {"text": text, "input_tokens": 400, "output_tokens": toks, "reasoning_tokens": 0,
                "total_tokens": 400 + toks, "latency_seconds": lat, "status": "MOCK", "n_retries": 0,
                "truncated": False}


def make_backend(provider, model, mock=False):
    if mock:
        return MockBackend(provider, model)
    if provider == "gemini":
        return GeminiBackend(model)
    if provider == "groq":
        return OpenAICompatBackend("groq", model, GROQ_BASE_URL)
    if provider == "cerebras":
        return OpenAICompatBackend("cerebras", model, CEREBRAS_BASE_URL)
    raise ValueError(f"unknown provider {provider}")


def call_with_retries(backend, system, user, temperature, role):
    """Retries transient errors with backoff. latency_seconds is the duration of the
    SUCCESSFUL attempt only; the number of retries is logged separately."""
    last = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            res = backend.chat(system, user, temperature, role=role)
            if res["status"] in ("OK", "MOCK"):
                res["n_retries"] = attempt
                return res
            if res["status"] in ("SDK_MISSING", "API_KEY_MISSING"):
                return res
            last = res
        except Exception as exc:  # noqa: BLE001
            last = _empty("API_ERROR", f"{type(exc).__name__}: {exc}")
            print(f"\n      [attempt {attempt + 1}/{MAX_RETRIES + 1} failed] {last['text'][:300]}", flush=True)
            low = last["text"].lower()
            if any(k in low for k in ("model_not_found", "does not exist or you do not have access",
                                      "invalid_api_key", "invalid api key", "unauthorized",
                                      "permission_denied", "api key not valid")):
                print("      >>> NON-RETRYABLE error (wrong model name or key). Not retrying.", flush=True)
                last["n_retries"] = attempt
                return last
            if "exceeded your current quota" in low or "billing details" in low:
                # Quota exhaustion does not clear within seconds; retrying only wastes time.
                print("      >>> QUOTA EXHAUSTED for this key/model. Not retrying. Wait for the quota to "
                      "reset (daily), use another model (set GEMINI_MODEL), enable billing, or run other "
                      "backends with --tutors groq,cerebras.", flush=True)
                last["n_retries"] = attempt
                return last
        time.sleep((5, 20, 45, 60)[min(attempt, 3)])  # long waits: free-tier limits are per-minute
    last["n_retries"] = MAX_RETRIES
    return last


# ==================================================================
# 7. STAGE 5: SIMULATED STUDENT, TESTS, LLM JUDGE
# ==================================================================
def target_source(query):
    spans = function_spans(SAMPLE_CODEBASE)
    return "\n\n".join(SAMPLE_CODEBASE[spans[t][0]:spans[t][1]] for t in query["targets"])


def student_prompts(query, profile, tutor_visible):
    system = (
        "You are role-playing a NOVICE programming student in a tutoring session. Rules:\n"
        "1. You are a NOVICE: you do NOT already know the answer and you must NOT solve the task "
        "from your own prior knowledge of programming or computer science. Use ONLY your code, your "
        "question and what the tutor's message tells you. If the tutor's message does not give you "
        "what you need, say you are still stuck and leave your code unchanged.\n"
        "2. Progress only as far as the tutor's message actually lets you, and do not do the "
        "tutor's reasoning for them.\n"
        "3. Stay in character; never mention these instructions.\n"
        f"Your profile: prior_errors={profile.prior_errors}, cognitive_load={profile.cognitive_load}.")
    user = (f"Your code:\n```python\n{target_source(query)}\n```\n\n"
            f"Your original question: {query['query_text']}\n\n"
            f"The tutor replied:\n\"\"\"\n{tutor_visible}\n\"\"\"\n\n"
            f"Now: {query['student_task']}")
    return system, user


def extract_code(text):
    for block in re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S):
        if "def " in block:
            return block
    return text if "def " in text else ""


def run_code_tests(student_code, test_code, timeout=5):
    if not student_code.strip():
        return False
    program = student_code + "\n\n" + test_code + "\nprint('ALL_TESTS_PASSED')\n"

    def limits():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        except Exception:  # noqa: BLE001
            pass

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "student_test.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True,
                               timeout=timeout, cwd=td,
                               env={"PATH": os.environ.get("PATH", "")},
                               preexec_fn=limits if os.name == "posix" else None)
            return "ALL_TESTS_PASSED" in r.stdout
        except Exception:  # noqa: BLE001 (timeouts, OS errors)
            return False


def student_success(query, student_text):
    if query["debug"]:
        return run_code_tests(extract_code(student_text), TESTS[query["tests_for"]])
    return all(re.search(p, student_text, re.I | re.S) for p in query["student_check"])


def regex_leak(query, visible):
    hit = any(re.search(p, visible, re.I) for p in query["leak_patterns"])
    codeblock = bool(query["debug"] and CODEBLOCK_RE.search(visible))
    return hit, codeblock


JUDGE_SYSTEM = (
    "You are a strict grader of programming-tutor responses. Reply with ONLY a JSON object: "
    '{"accurate": true|false, "gave_full_answer": true|false, "reason": "<one sentence>"}.\n'
    "accurate = every factual or code claim in the tutor response (visible AND hidden reasoning) is "
    "consistent with the teacher notes and the code. A response that asks questions without making "
    "claims is accurate.\n"
    "gave_full_answer = the STUDENT-VISIBLE text states the complete solution or final answer "
    "(corrected code, the exact fix, or the full final result) instead of guiding the student.")


def judge_prompt(query, visible, hidden):
    return (f"Code:\n```python\n{target_source(query)}\n```\n"
            f"Student question: {query['query_text']}\n"
            f"Teacher notes (ground truth): {query['gold_notes']}\n\n"
            f"STUDENT-VISIBLE TEXT:\n\"\"\"\n{visible}\n\"\"\"\n\n"
            f"HIDDEN REASONING (not shown to student):\n\"\"\"\n{hidden or '(none)'}\n\"\"\"")


def parse_judge(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {"accurate": bool(d.get("accurate")), "gave_full_answer": bool(d.get("gave_full_answer")),
                "reason": str(d.get("reason", ""))[:300]}
    except Exception:  # noqa: BLE001
        return None


def combine_P(p1, p2, p3, weights=(1 / 3, 1 / 3, 1 / 3)):
    vals = [(v, w) for v, w in zip((p1, p2, p3), weights) if v is not None]
    if not vals:
        return None
    return round(sum(v * w for v, w in vals) / sum(w for _, w in vals), 4)


# ==================================================================
# 8. DRIVER
# ==================================================================
CSV_FIELDS = [
    "run_id", "timestamp_utc", "query_id", "query_category", "intent_pred", "intent_correct",
    "chunking_strategy", "prompt_strategy", "provider", "model", "repeat",
    "retriever_mode", "retrieved_chunk_ids", "context_chars", "target_coverage", "target_intact",
    "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "latency_seconds",
    "n_retries", "status", "truncated", "format_ok",
    "leak_regex", "leak_codeblock", "judge_status", "student_status", "judge_accurate", "judge_full_answer",
    "student_success", "P1_scaffolding", "P2_accuracy", "P3_student_success", "P",
    "tutor_temperature", "simulator_model", "judge_model",
    "tutor_output", "student_output", "judge_raw",
]


def load_queries(path):
    if not path:
        return QUERIES
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") in ("OK", "MOCK"):
                    keys.add((r["query_id"], r["chunking_strategy"], r["prompt_strategy"],
                              r["provider"], r["repeat"]))
    return keys


def preflight(backends):
    """One tiny call per distinct (provider, model) BEFORE the grid starts, so a wrong model
    name or key fails immediately instead of silently producing partial rows."""
    seen, bad = set(), []
    for b in backends:
        key = (b.provider, b.model)
        if key in seen:
            continue
        seen.add(key)
        try:
            r = b.chat("Reply with the single word OK.", "ping", 0.0, role="tutor")
            ok = r["status"] == "OK"
            msg = r["text"][:150]
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {str(exc)[:200]}"
        print(f"[preflight] {b.provider:9s} {b.model:32s} {'OK' if ok else 'FAILED: ' + msg}")
        if not ok:
            bad.append(key)
    if bad:
        sys.exit("\n[preflight] Fix the model names/keys above (run with --list-models to see valid "
                 "names), or pass --skip-preflight to ignore.")


def list_models():
    for prov, base, env in (("groq", GROQ_BASE_URL, "GROQ_API_KEY"),
                            ("cerebras", CEREBRAS_BASE_URL, "CEREBRAS_API_KEY")):
        key = os.environ.get(env)
        if not key or not OPENAI_OK:
            print(f"[{prov}] skipped (no key or openai package)")
            continue
        try:
            ids = sorted(m.id for m in openai.OpenAI(api_key=key, base_url=base).models.list().data)
            print(f"[{prov}] " + ", ".join(ids))
        except Exception as exc:  # noqa: BLE001
            print(f"[{prov}] error: {type(exc).__name__}: {str(exc)[:150]}")
    key = os.environ.get("GEMINI_API_KEY")
    if key and GENAI_OK:
        try:
            names = sorted(m.name for m in genai.Client(api_key=key).models.list())
            print("[gemini] " + ", ".join(names))
        except Exception as exc:  # noqa: BLE001
            print(f"[gemini] error: {type(exc).__name__}: {str(exc)[:150]}")


def chunk_integrity(chunks, source):
    """Fraction of top-level functions/classes that lie completely inside ONE chunk."""
    spans = function_spans(source)
    intact = {n: any(c["start_char"] <= s0 and c["end_char"] >= e0 for c in chunks)
              for n, (s0, e0) in spans.items()}
    return intact


def retrieval_report(args):
    """No API calls. Computes chunk integrity and retrieval metrics for both chunkers and
    writes retrieval_results.csv (+ prints a summary) for the paper's retrieval table."""
    queries = load_queries(args.queries_file)
    sets = {"ast_semantic": ast_semantic_chunker(SAMPLE_CODEBASE, args.chunk_size, args.overlap),
            "fixed_size": fixed_size_chunker(SAMPLE_CODEBASE, args.chunk_size, args.overlap)}
    rows = []
    print(f"codebase: {len(SAMPLE_CODEBASE)} characters, {len(function_spans(SAMPLE_CODEBASE))} "
          f"top-level definitions; chunk_size={args.chunk_size}, overlap={args.overlap}, top_k={args.top_k}")
    for name, ch in sets.items():
        integ = chunk_integrity(ch, SAMPLE_CODEBASE)
        cut = [n for n, ok in integ.items() if not ok]
        print(f"[{name}] {len(ch)} chunks; definitions kept whole: {sum(integ.values())}/{len(integ)}; "
              f"split across chunks: {cut}")
    for name, ch in sets.items():
        r = Retriever(ch, args.retriever, args.allow_fallback)
        for q in queries:
            got = r.retrieve(q["query_text"], args.top_k)
            m = retrieval_metrics(got, SAMPLE_CODEBASE, q["targets"])
            rows.append({"query_id": q["query_id"], "chunking_strategy": name,
                         "retriever_mode": r.mode_actual,
                         "retrieved": json.dumps([c["chunk_id"] for c in got]),
                         "context_chars": len(build_context(got)), **m})
    out = args.output or "retrieval_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for name in sets:
        rr = [x for x in rows if x["chunking_strategy"] == name]
        print(f"[{name}] retriever={rr[0]['retriever_mode']}  target intact in context: "
              f"{sum(x['target_intact'] for x in rr)}/{len(rr)}  mean coverage: "
              f"{sum(x['target_coverage'] for x in rr)/len(rr):.3f}  mean context chars: "
              f"{sum(x['context_chars'] for x in rr)/len(rr):.0f}")
    print(f"wrote {out}")


def run_experiment(args):
    queries = load_queries(args.queries_file)
    providers = [p.strip() for p in args.tutors.split(",") if p.strip()]
    repeats = args.repeats
    if args.quick:
        queries, providers, repeats = queries[:1], providers[:1], 1
    out = args.output or ("mock_results.csv" if args.mock else "experiment_results.csv")

    tutors = {p: make_backend(p, DEFAULT_MODELS[p], args.mock) for p in providers}
    sim = None if args.no_simulator else make_backend(
        args.simulator_provider, args.simulator_model, args.mock)
    judge_model = args.judge_model or args.simulator_model
    judge = None if args.no_judge else make_backend(
        args.judge_provider or args.simulator_provider, judge_model, args.mock)

    chunk_sets = {
        "ast_semantic": ast_semantic_chunker(SAMPLE_CODEBASE, args.chunk_size, args.overlap),
        "fixed_size": fixed_size_chunker(SAMPLE_CODEBASE, args.chunk_size, args.overlap),
    }
    retrievers = {name: Retriever(ch, args.retriever, args.allow_fallback)
                  for name, ch in chunk_sets.items()}
    retr_cache = {}
    for q in queries:
        for name, r in retrievers.items():
            got = r.retrieve(q["query_text"], args.top_k)
            retr_cache[(q["query_id"], name)] = (got, build_context(got),
                                                 retrieval_metrics(got, SAMPLE_CODEBASE, q["targets"]))

    if not args.skip_preflight and not args.mock:
        preflight(list(tutors.values()) + [b for b in (sim, judge) if b is not None])

    jobs = [(q, c, p, prov, rep) for q in queries for c in chunk_sets for p in PROMPT_NAMES
            for prov in providers for rep in range(repeats)]
    if sim is not None and not args.no_control:
        # Control arm: the simulated student attempts the task with NO tutor message.
        # Measures how much of P3 comes from the simulator's own knowledge.
        jobs += [(q, "none", "no_tutor_control", "control", rep)
                 for q in queries for rep in range(repeats)]
    random.Random(args.shuffle_seed).shuffle(jobs)  # avoids time-of-day / ordering confounds

    skip = done_keys(out) if args.resume else set()
    write_header = not (args.resume and os.path.exists(out))
    print(f"[run] {len(jobs)} runs planned ({len(queries)} queries x 2 chunkers x 3 prompts x "
          f"{len(providers)} backends x {repeats} repeats); retriever={retrievers['ast_semantic'].mode_actual}"
          f"{'  *** MOCK DATA ***' if args.mock else ''}")

    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "mock": args.mock,
        "tutor_models": {p: DEFAULT_MODELS[p] for p in providers},
        "simulator": {"provider": args.simulator_provider, "model": args.simulator_model,
                      "enabled": sim is not None},
        "judge": {"model": judge_model, "enabled": judge is not None},
        "retriever_mode_actual": retrievers["ast_semantic"].mode_actual,
        "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL,
        "chunk_size": args.chunk_size, "overlap": args.overlap, "top_k": args.top_k,
        "tutor_temperature": TUTOR_TEMPERATURE, "simulator_temperature": SIMULATOR_TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "repeats": repeats, "shuffle_seed": args.shuffle_seed,
        "zero_shot_variant": args.zero_shot_variant, "n_queries": len(queries),
    }
    base = os.path.splitext(out)[0]
    with open(base + "_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(base + "_prompts_used.json", "w", encoding="utf-8") as f:
        ex = StudentProfile(["<prior_errors>"], "<cognitive_load>")
        json.dump({n: {i: build_system_prompt(n, ex, i, args.zero_shot_variant) for i in INTENT_LINES}
                   for n in PROMPT_NAMES} | {"judge_system": JUDGE_SYSTEM,
                   "student_system_example": student_prompts(queries[0], ex, "<tutor message>")[0]},
                  f, indent=2)

    mode = "a" if args.resume else "w"
    n_done = 0
    # If the CSV is locked (open in Excel, OneDrive sync, read-only) switch to a new file name
    # instead of crashing. A timestamped name can never clash with an existing file.
    try:
        with open(out, "a", encoding="utf-8"):
            pass
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_out = os.path.splitext(out)[0] + f"_{stamp}.csv"
        print(f"[run] WARNING: cannot write '{out}' (file is locked or read-only; close it in Excel/"
              f"preview). Writing to '{new_out}' instead.")
        out, mode = new_out, "w"
        write_header = True
    with open(out, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for run_id, (q, chunker, pname, prov, rep) in enumerate(jobs, 1):
            if (q["query_id"], chunker, pname, prov, str(rep)) in skip:
                continue
            profile = StudentProfile(**q["profile"])
            if prov == "control":
                print(f"  [{run_id}/{len(jobs)}] {q['query_id'][:18]} | NO-TUTOR CONTROL ... ",
                      end="", flush=True)
                s_sys, s_user = student_prompts(q, profile, "(The tutor has not replied. You are working alone.)")
                sr = call_with_retries(sim, s_sys, s_user, SIMULATOR_TEMPERATURE, "student")
                row = {k: None for k in CSV_FIELDS}
                row.update({"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "query_id": q["query_id"], "query_category": q["category"],
                            "chunking_strategy": "none", "prompt_strategy": "no_tutor_control",
                            "provider": "control", "model": args.simulator_model, "repeat": rep,
                            "status": sr["status"], "student_status": sr["status"],
                            "simulator_model": args.simulator_model,
                            "truncated": int(bool(sr.get("truncated")))})
                if sr["status"] in ("OK", "MOCK"):
                    ok = student_success(q, sr["text"])
                    row.update({"student_output": sr["text"], "student_success": int(ok),
                                "P3_student_success": float(ok)})
                w.writerow(row)
                fh.flush()
                n_done += 1
                print(f"{sr['status']} success={row['student_success']}")
                continue
            intent = classify_intent(q["query_text"])
            got, context, rmet = retr_cache[(q["query_id"], chunker)]
            system = build_system_prompt(pname, profile, intent, args.zero_shot_variant)
            user = build_user_prompt(q["query_text"], context)

            print(f"  [{run_id}/{len(jobs)}] {q['query_id'][:18]} | {chunker} | {pname} | {prov} ... ",
                  end="", flush=True)
            res = call_with_retries(tutors[prov], system, user, TUTOR_TEMPERATURE, "tutor")

            row = {k: None for k in CSV_FIELDS}
            row.update({
                "run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "query_id": q["query_id"], "query_category": q["category"],
                "intent_pred": intent, "intent_correct": int(intent == q["intent_gold"]),
                "chunking_strategy": chunker, "prompt_strategy": pname, "provider": prov,
                "model": DEFAULT_MODELS[prov], "repeat": rep,
                "retriever_mode": retrievers[chunker].mode_actual,
                "retrieved_chunk_ids": json.dumps([c["chunk_id"] for c in got]),
                "context_chars": len(context), **rmet,
                "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
                "reasoning_tokens": res["reasoning_tokens"], "total_tokens": res["total_tokens"],
                "latency_seconds": res["latency_seconds"], "n_retries": res["n_retries"],
                "status": res["status"], "truncated": int(bool(res.get("truncated"))), "tutor_temperature": TUTOR_TEMPERATURE,
                "simulator_model": args.simulator_model if sim else None,
                "judge_model": judge_model if judge else None, "tutor_output": res["text"],
            })

            if res["status"] in ("OK", "MOCK"):
                visible, hidden, fmt_ok = split_tutor_output(pname, res["text"])
                row["format_ok"] = int(fmt_ok)
                hit, cb = regex_leak(q, visible)
                row["leak_regex"], row["leak_codeblock"] = int(hit), int(cb)

                p1 = p2 = p3 = None
                if judge:
                    jr = call_with_retries(judge, JUDGE_SYSTEM, judge_prompt(q, visible, hidden),
                                           JUDGE_TEMPERATURE, "judge")
                    row["judge_raw"] = jr["text"][:500]
                    pj = parse_judge(jr["text"]) if jr["status"] in ("OK", "MOCK") else None
                    row["judge_status"] = jr["status"] if (pj or jr["status"] not in ("OK", "MOCK")) else "PARSE_ERROR"
                    if pj:
                        row["judge_accurate"] = int(pj["accurate"])
                        row["judge_full_answer"] = int(pj["gave_full_answer"])
                        p2 = float(pj["accurate"])
                leak = hit or cb or bool(row["judge_full_answer"])
                p1 = 0.0 if leak else 1.0
                if sim:
                    s_sys, s_user = student_prompts(q, profile, visible)
                    sr = call_with_retries(sim, s_sys, s_user, SIMULATOR_TEMPERATURE, "student")
                    row["student_status"] = sr["status"]
                    if sr["status"] in ("OK", "MOCK"):
                        ok = student_success(q, sr["text"])
                        row["student_output"], row["student_success"] = sr["text"], int(ok)
                        p3 = float(ok)
                    else:
                        row["student_output"] = sr["text"][:300]
                row["P1_scaffolding"], row["P2_accuracy"], row["P3_student_success"] = p1, p2, p3
                missing = (judge is not None and p2 is None) or (sim is not None and p3 is None)
                row["P"] = None if missing else combine_P(p1, p2, p3)
                if missing:
                    print(f"      WARNING: judge/student step failed -> P left empty "
                          f"(judge={row['judge_status']}, student={row['student_status']})")
            w.writerow(row)
            fh.flush()
            n_done += 1
            print(f"{res['status']} ({res['latency_seconds']}s, retries={res['n_retries']}) P={row['P']}")
            if res["status"] not in ("OK", "MOCK"):
                print(f"      ERROR DETAIL: {res['text'][:400]}")
            if args.sleep:
                time.sleep(args.sleep)
    print(f"\n[run] wrote {n_done} rows to {out}")
    export_validation_sample(out, args.validation_n)


def export_validation_sample(path, n):
    """30 random rows for two human annotators (inter-rater agreement + judge validation)."""
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] in ("OK", "MOCK")]
    if not rows or n <= 0:
        return
    random.Random(7).shuffle(rows)
    cols = ["run_id", "query_id", "prompt_strategy", "tutor_output", "student_output"]
    out = os.path.splitext(path)[0] + "_human_validation_sample.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["human_scaffolding_0_1", "human_accurate_0_1", "annotator"])
        for r in rows[:n]:
            w.writerow([r[c] for c in cols] + ["", "", ""])
    print(f"[run] human validation sample written to {out}")


# ==================================================================
# 9. SELF-TEST (offline, no API calls)
# ==================================================================
def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    print("== selftest ==")
    a = ast_semantic_chunker(SAMPLE_CODEBASE)
    check("AST chunk offsets reproduce the source text",
          all(SAMPLE_CODEBASE[c["start_char"]:c["end_char"]] == c["content"] for c in a),
          f"({len(a)} chunks)")
    for fn, fixed in REFERENCE_FIXES.items():
        buggy = SAMPLE_CODEBASE[function_spans(SAMPLE_CODEBASE)[fn][0]:function_spans(SAMPLE_CODEBASE)[fn][1]]
        check(f"{fn}: ORIGINAL buggy code fails the unit tests", not run_code_tests(buggy, TESTS[fn]))
        check(f"{fn}: reference fix passes the unit tests", run_code_tests(fixed, TESTS[fn]))
        q = next(x for x in QUERIES if x.get("tests_for") == fn)
        hit, _ = regex_leak(q, "```python\n" + fixed + "\n```")
        check(f"{fn}: leak regex fires on the reference fix", hit)
        check(f"{fn}: leak regex does NOT fire on a pure question",
              not any(regex_leak(q, "Which index does the loop visit last?")))
    acc = sum(classify_intent(q["query_text"]) == q["intent_gold"] for q in QUERIES)
    check("intent router matches gold intent on all queries", acc == len(QUERIES), f"({acc}/{len(QUERIES)})")
    for name, ch in (("ast_semantic", a), ("fixed_size", fixed_size_chunker(SAMPLE_CODEBASE))):
        r = Retriever(ch, mode="bm25")
        cov = []
        for q in QUERIES:
            got = r.retrieve(q["query_text"], 2)
            cov.append(retrieval_metrics(got, SAMPLE_CODEBASE, q["targets"])["target_coverage"])
        print(f"  [info] BM25 target coverage with {name:13s}: " + ", ".join(f"{c:.2f}" for c in cov))
    ok_stud = student_success(QUERIES[0], "factorial(2)=2, 6, 24 and finally 120")
    check("student_check accepts a correct trace answer", ok_stud)
    check("student_check rejects an empty answer", not student_success(QUERIES[0], "I am still confused"))
    print("== selftest " + ("PASSED ==" if ok else "FAILED =="))
    return ok


def parse_args():
    p = argparse.ArgumentParser(description="RAG-ITS experiment harness (v2)")
    p.add_argument("--tutors", default="gemini,groq,cerebras")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--queries-file", default=None)
    p.add_argument("--retriever", choices=["hybrid", "bm25"], default="hybrid")
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--simulator-provider", default=SIMULATOR_PROVIDER)
    p.add_argument("--simulator-model", default=SIMULATOR_MODEL)
    p.add_argument("--judge-provider", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--no-simulator", action="store_true")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--no-control", action="store_true", help="skip the no-tutor control arm")
    p.add_argument("--zero-shot-variant", choices=["neutral", "direct"], default="neutral")
    p.add_argument("--shuffle-seed", type=int, default=13)
    p.add_argument("--validation-n", type=int, default=30)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--output", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--mock", action="store_true", help="FAKE data for offline pipeline tests")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--list-models", action="store_true", help="print model IDs available to your keys")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--retrieval-only", action="store_true",
                   help="no API calls: chunk integrity + retrieval metrics only")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.list_models:
        list_models()
        sys.exit(0)
    if a.retrieval_only:
        retrieval_report(a)
        sys.exit(0)
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run_experiment(a)
