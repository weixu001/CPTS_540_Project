#!/usr/bin/env python
"""
build_ivf_hnsw.py

Build a *hybrid* IVF-Flat + HNSW FAISS index (inner product) with a
safe GPU‐conversion step.
"""
import argparse, glob, numpy as np, faiss, torch

def build_ivf_hnsw(vectors: np.ndarray,
                   nlist: int,
                   nprobe: int,
                   M: int,
                   use_gpu: bool=True,
                   gpu_id: int=0):
    d = vectors.shape[1]

    # 1) HNSW quantizer
    quantizer = faiss.IndexHNSWFlat(d, M)
    quantizer.hnsw.efConstruction = 200
    quantizer.metric_type = faiss.METRIC_INNER_PRODUCT

    # 2) IVF-Flat on top of that
    ivf = faiss.IndexIVFFlat(
        quantizer,
        d,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )
    ivf.nprobe = nprobe

    # 3) train & add
    print(f"⏳ training IVF-Flat (with HNSW quantizer) on {vectors.shape[0]} vectors")
    ivf.train(vectors)
    print(f"➕ adding {vectors.shape[0]} vectors")
    ivf.add(vectors)

    # 4) optionally move to GPU, but guard against failures
    if use_gpu and torch.cuda.is_available():
        print(f"🚚 attempting to move index to GPU {gpu_id} …")
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False
        co.allowCpuCoarseQuantizer = True

        # stash a CPU copy in case GPU copy fails
        cpu_ivf = ivf
        try:
            ivf = faiss.index_cpu_to_gpu(res, gpu_id, ivf, co)
            # GPU copy succeeded; free the CPU‐only reference
            del cpu_ivf
            print("✅ moved index to GPU successfully")
        except Exception as e:
            print(f"⚠️  GPU conversion failed, continuing with CPU index: {e}")
            ivf = cpu_ivf

    return ivf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_glob",  required=True)
    p.add_argument("--meta_glob", required=True)
    p.add_argument("--index_out", required=True)
    p.add_argument("--meta_out",  required=True)
    p.add_argument("--nlist",     type=int, default=4096)
    p.add_argument("--nprobe",    type=int, default=64)
    p.add_argument("--M",         type=int, default=32)
    p.add_argument("--gpu_id",    type=int, default=0)
    p.add_argument("--no_gpu",    action="store_true")
    args = p.parse_args()

    # load all embedding shards
    emb_files = sorted(glob.glob(args.emb_glob))
    all_embs  = np.vstack([np.load(f) for f in emb_files]).astype("float32")
    faiss.normalize_L2(all_embs)

    # load all metadata shards
    meta_files = sorted(glob.glob(args.meta_glob))
    all_meta   = np.concatenate([np.load(f, allow_pickle=True) for f in meta_files])

    use_gpu = not args.no_gpu
    idx = build_ivf_hnsw(
        all_embs,
        nlist   = args.nlist,
        nprobe  = args.nprobe,
        M       = args.M,
        use_gpu = use_gpu,
        gpu_id  = args.gpu_id
    )

    # 5) persist to disk
    print(f"writing index to {args.index_out}")
    cpu_index = faiss.index_gpu_to_cpu(idx) if (use_gpu and torch.cuda.is_available()) else idx
    faiss.write_index(cpu_index, args.index_out)

    print(f"writing metadata to {args.meta_out}")
    np.save(args.meta_out, all_meta)

    print("Done.")

if __name__=="__main__":
    main()
