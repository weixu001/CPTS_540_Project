#!/usr/bin/env python3
"""
Build a ‘pre-retrieved’ TriviaQA file for generator fine-tuning.

Each JSONL line:
{
  "question": "...",
  "answers":  ["gold", "aliases", ...],
  "contexts": [
      {"title": ..., "score": 12.3},
      ...
  ]
}
"""
import argparse, json, time
from pathlib import Path

import numpy as np, faiss, torch
from datasets import load_dataset
from transformers import (
    DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
)

# ───────────────────────────────────────────────────────────────
def dpr_embed(questions, tok, enc, device):
    batch = tok(
        questions,
        padding=True, truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        return enc(**batch).pooler_output.cpu().numpy()

# ───────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_dir",      required=True)
    pa.add_argument("--index_file",    required=True)
    pa.add_argument("--meta_file",     required=True)
    pa.add_argument("--output_json",   default="pretrieved_triviaqa.jsonl")
    pa.add_argument("--k",             type=int, default=5)
    pa.add_argument("--split",         type=float, default=0.8)
    pa.add_argument("--max_samples",   type=int, default=None)
    pa.add_argument("--batch_size",    type=int, default=32)
    args = pa.parse_args()

    # 1) FAISS index + metadata
    print("🔹 loading FAISS index:", args.index_file)
    index = faiss.read_index(args.index_file)
    meta  = np.load(args.meta_file, allow_pickle=True)
    print("    ntotal:", index.ntotal)

    # 2) DPR question encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_tok  = DPRQuestionEncoderTokenizerFast.from_pretrained(
               "facebook/dpr-question_encoder-single-nq-base")
    q_enc  = DPRQuestionEncoder.from_pretrained(
               "facebook/dpr-question_encoder-single-nq-base").to(device).eval()

    # 3) TriviaQA
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
    if args.max_samples:
        ds = ds.select(range(args.max_samples))
    n = len(ds)
    print(f"🔹 sampling {n} examples")

    # 4) iterate & write JSONL
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.output_json, "w")
    t0 = time.time()

    for start in range(0, n, args.batch_size):
        batch = ds[start : start + args.batch_size]

        # batch is a dict of columns → list
        questions = [q.strip().replace("\n", " ") for q in batch["question"]]
        answers = [
            (a["aliases"] or [a["value"]])
            for a in batch["answer"]
        ]

        q_vecs = dpr_embed(questions, q_tok, q_enc, device)
        D, I   = index.search(q_vecs, args.k)

        for q, golds, dists, ids in zip(questions, answers, D, I):
            ctxs = [str(meta[idx]) for idx in ids]
            out.write(json.dumps({
                "question": q,
                "answer":  golds[0],
                "contexts": ctxs
            }) + "\n")

        if (start // args.batch_size) % 10 == 0:
            done = min(start + args.batch_size, n)
            print(f"  processed {done}/{n}", end="\r")

    out.close()
    print(f"\n✅ wrote {n} lines to {args.output_json} "
          f"({time.time() - t0:.1f}s)")

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
