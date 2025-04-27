#!/usr/bin/env python
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

def main():
    parser = argparse.ArgumentParser(
        description="Embed a Wikipedia shard with DPR and dump two .npy files"
    )
    parser.add_argument("--subset",    type=str, default="20220301.en",
                        help="which Wikipedia dump to use")
    parser.add_argument("--shard_id",  type=int, required=True,
                        help="this shard’s ID (0..num_shards-1)")
    parser.add_argument("--num_shards",type=int, required=True,
                        help="total number of shards")
    parser.add_argument("--chunk_size",type=int, default=100,
                        help="max tokens per chunk")
    parser.add_argument("--stride",    type=int, default=0,
                        help="chunk overlap in tokens")
    parser.add_argument("--index_out", type=str, required=True,
                        help="where to write the embeddings .npy")
    parser.add_argument("--meta_out",  type=str, required=True,
                        help="where to write the metadata .npy")
    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load your shard of Wikipedia
    wiki = load_dataset("wikipedia", args.subset, split="train", trust_remote_code=True)
    N = len(wiki)
    per = N // args.num_shards
    start = args.shard_id * per
    end = (args.shard_id+1)*per if args.shard_id < args.num_shards-1 else N
    print(f"Shard {args.shard_id}/{args.num_shards}: articles [{start}:{end})")

    # DPR encoders
    tok = DPRContextEncoderTokenizerFast.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base", use_fast=True
    )
    enc = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    ).to(device).eval()

    all_embs = []
    all_meta = []

    for i in range(start, end):
        text = wiki[i]["text"]
        toks = tok(
            text,
            truncation=True, max_length=args.chunk_size,
            stride=args.stride,
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = enc(**toks).pooler_output.cpu().numpy()

        ids = toks["input_ids"].cpu().numpy()
        for j in range(out.shape[0]):
            all_embs.append(out[j])
            all_meta.append(tok.decode(ids[j], skip_special_tokens=True))

    # Stack & save
    embs = np.vstack(all_embs)
    np.save(args.index_out, embs)
    np.save(args.meta_out, np.array(all_meta, dtype=object))
    print(f"Shard {args.shard_id} → {args.index_out}, {args.meta_out}")

if __name__=="__main__":
    main()
