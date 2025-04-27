# Retrieval‑Augmented QA on Wikipedia  
Dense DPR index · T5‑LoRA generator · optional RAG fine‑tuning (Kamiak HPC)

---

## 0. Repository layout  

```
CPTS_540_Project
├── data/                     
│   ├── wiki_emb_shard*.npy
│   └── wiki_emb_meta_shard*.npy
├── src/
│   ├── wiki_index.py                
│   ├── wiki_hnsw_index.py        
│   ├── wiki_build_passages.py        
│   ├── wiki_prep_retrieval.py       
│   ├── train_generator.py           
│   ├── evaluation.py                
├── slurm/
│   └── submit_RAG.sh   
├── logs/                       
└── README.md                    
```

---

## 1. Environment

```bash
module load miniconda3/3.12
conda create -n triviaqa python=3.9 -y
conda activate triviaqa

pip install "transformers>=4.51" datasets accelerate faiss-gpu peft \
            sentencepiece evaluate pytorch-lightning

# Optional: Kamiak scratch cache
export HF_HOME=$SCRATCH/hf_home
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
```

---

## 2. End‑to‑end pipeline

Submit the job:

```bash
sbatch submit_RAG.sh
```

The script executes:

| Step | Script           | Output |
|------|----------------------------|-----------------|
| 0|  wiki_index.py  |'download wikipedia dump'|
| 1 | wiki_hnsw_index.py        | `wiki_hnsw.faiss`, `wiki_hnsw_meta.npy` |
| 2 | wiki_build_passages.py     | `wiki_passages_dataset_hnsw/` |
| 3 | wiki_prep_retrieval.py         | `pretrieved_triviaqa_hnsw.jsonl` |
| 4 | train_generator.py                 | `t5_lora_ft/`  |
| 5 | finetune_rag.py           | `rag_end2end_ft/` (generator + retriever) |
| 6 | evaluation.py          | JSON metrics + throughput stats |

All files stay under the Slurm scratch workspace of the job.

---

## 3. Model configurations (×6)

| `--mode`   | Model / strategy                            | Retrieval style |
|------------|---------------------------------------------|-----------------|
| base  | T5‑base + DPR                                | single‑shot top‑k |
| lora   | T5‑LoRA + DPR                                | single‑shot top‑k |
| rerank | T5‑base + DPR → cross‑encoder rerank         | top‑k → rerank‑n |
|iter   | *Iterative* DPR →T5‑base                    | 2 iterations |
| gen_only | T5‑base without context               | — |
| rag    | RAG‑token‑base, end‑to‑end fine‑tuned        | built‑in |

Evaluate all modes on 12000 TriviaQA‑validation questions:

```bash
python src/evaluate_pipeline.py --mode all --sample_size 12000 --k 15 \
       --lora_dir t5_lora_ft \
       --ckpt_dir rag_end2end_ft/checkpoint-best \
       --passages_path wiki_passages_dataset_hnsw \
       --index_path wiki_passages_dataset_hnsw/embeddings.faiss
```

###4. Results on **TriviaQA‑validation** (12000 queries)

| mode | Recall@15 | EM | F1 | QPS |
|------|-----------|----|----|-----|
| gen_only (No RAG) | 0.000 | 0.0005 | 0.0024 | **8.2** |
| base | 0.609 | 0.2243 | 0.2996 | 6.0 |
| lora | 0.609 | 0.1293 | 0.1892 | 5.6 |
| rerank | 0.609 | 0.1508 | 0.2142 | 2.7 |
| iter | 0.609 | 0.1536 | 0.2250 | 3.7 |
| rag (end‑to‑end) | 0.000 | 0.0780 | 0.1457 | 7.3 |






