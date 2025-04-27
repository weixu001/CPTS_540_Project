#!/usr/bin/env python3
import glob, os, shutil
import numpy as np
from datasets import Dataset

def gen_passages(shard_emb_glob, shard_meta_glob):
    emb_files  = sorted(glob.glob(shard_emb_glob))
    meta_files = sorted(glob.glob(shard_meta_glob))
    assert len(emb_files) == len(meta_files), "mismatched shards"
    idx = 0
    for emb_f, meta_f in zip(emb_files, meta_files):
        embs  = np.load(emb_f).astype("float32")
        metas = np.load(meta_f, allow_pickle=True)
        for text, emb in zip(metas, embs):
            yield {
                "id":         idx,         # optional but recommended
                "title":      "",
                "text":       text,
                "embeddings": emb.tolist(),
            }
            idx += 1

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Build HF arrow dataset of passages + embeddings"
    )
    p.add_argument("--shard_emb_glob",  default="wiki_emb_shard*.npy")
    p.add_argument("--shard_meta_glob", default="wiki_emb_meta_shard*.npy")
    p.add_argument("--out_dir",         required=True,
                   help="where to save the HF dataset")
    p.add_argument("--index_file",      required=True,
                   help="your prebuilt .faiss index")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # stream into HF Dataset
    ds = Dataset.from_generator(
        gen_passages,
        gen_kwargs={
            "shard_emb_glob":  args.shard_emb_glob,
            "shard_meta_glob": args.shard_meta_glob,
        }
    )
    ds.save_to_disk(args.out_dir)
    print(f"✅ HF dataset saved to {args.out_dir}")

    # copy the FAISS index alongside
    dst = os.path.join(args.out_dir, "embeddings.faiss")
    shutil.copy(args.index_file, dst)
    print(f"✅ .faiss index copied to {dst}")
