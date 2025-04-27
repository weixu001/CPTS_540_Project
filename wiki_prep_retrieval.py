#!/usr/bin/env python3
"""
Create train/val/test splits of Question → Answer for RAG end2end
and write out:
  train.source  (questions)
  train.target  (gold answers)
  val.source
  val.target
  test.source
  test.target
"""
import os
import random
from datasets import load_dataset

def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Build question/answer .source/.target files for RAG end2end"
    )
    p.add_argument("--data_dir",      required=True,
                   help="where to write .source and .target files")
    p.add_argument("--split",  type=float, default=0.8,
                   help="fraction for train; rest equally val/test")
    p.add_argument("--seed",   type=int,   default=42)
    p.add_argument("--max_samples", type=int, default=None,
                   help="if set, only take this many questions")
    args = p.parse_args()

    # 1) load TriviaQA
    ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
    if args.max_samples:
        ds = ds.select(range(args.max_samples))

    # 2) extract (question, answer) pairs
    qa = []
    for ex in ds:
        q = ex["question"].strip().replace("\n", " ")
        # take one gold answer
        golds = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        a = golds[0].strip().replace("\n", " ")
        qa.append((q, a))

    # 3) shuffle & split
    random.seed(args.seed)
    random.shuffle(qa)
    n_train = int(len(qa) * args.split)
    n_rest  = len(qa) - n_train
    n_val   = n_rest // 2

    splits = {
        "train": qa[:n_train],
        "val":   qa[n_train:n_train + n_val],
        "test":  qa[n_train + n_val:]
    }

    # 4) write out .source / .target for each split
    os.makedirs(args.data_dir, exist_ok=True)
    for name, items in splits.items():
        src_path = os.path.join(args.data_dir, f"{name}.source")
        tgt_path = os.path.join(args.data_dir, f"{name}.target")
        with open(src_path, "w") as f_q, open(tgt_path, "w") as f_a:
            for q, a in items:
                f_q.write(q + "\n")
                f_a.write(a + "\n")
        print(f"→ Wrote {len(items)} lines to {src_path} & {tgt_path}")

if __name__ == "__main__":
    main()
