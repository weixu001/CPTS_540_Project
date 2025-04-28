#!/bin/bash
#SBATCH --job-name=RAGproject
#SBATCH --partition=camas
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rag_project_%j.out
#SBATCH --error=logs/rag_project_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weikun.xu@wsu.edu
set -euo pipefail

# ---------------------------------------------------------------------
# 0) Per-job scratch workspace
# ---------------------------------------------------------------------
myscratch=$(mkworkspace -n job_${SLURM_JOBID})
echo "→ Scratch workspace = $myscratch"
cd   "$myscratch"

# ---------------------------------------------------------------------
# 1) HuggingFace caches inside scratch
# ---------------------------------------------------------------------
export HF_HOME="$myscratch/hf_home"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
echo "✔ HF_HOME            = $HF_HOME"
echo "✔ HF_DATASETS_CACHE  = $HF_DATASETS_CACHE"
echo "✔ TRANSFORMERS_CACHE = $TRANSFORMERS_CACHE"

# ---------------------------------------------------------------------
# 2) Load Conda env
# ---------------------------------------------------------------------
module load miniconda3/3.12
eval "$(conda shell.bash hook)"
conda env create -f environment.yml
conda activate triviaqa

# ---------------------------------------------------------------------
# 3) Where is the repo?
#    (clone once to $SCRATCH, then reuse)
# ---------------------------------------------------------------------
REPO="$SCRATCH/CPTS_540_Project"         
SRC="$REPO/src"                          
RAG="$REPO/transformers-research-projects/rag-end2end-retriever"

# ---------------------------------------------------------------------
# 4) Download the Wikipedia Dump + Build IVF-HNSW index  ➜  wiki_hnsw.faiss + meta
# ---------------------------------------------------------------------
python "$SRC/wiki_index.py" \
    --subset       20220301.en \
    --shard_id     $SLURM_ARRAY_TASK_ID \
    --num_shards   10 \
    --chunk_size   100 \
    --stride       0 \
    --index_out    "$REPO/data/wiki_emb_shard${SLURM_ARRAY_TASK_ID}.npy" \
    --meta_out     "$REPO/data/wiki_emb_meta_shard${SLURM_ARRAY_TASK_ID}.npy"


python "$SRC/wiki_hnsw_index.py" \
  --emb_glob   "$REPO/data/wiki_emb_shard*.npy" \
  --meta_glob  "$REPO/data/wiki_emb_meta_shard*.npy" \
  --index_out  "wiki_hnsw.faiss" \
  --meta_out   "wiki_hnsw_meta.npy" \
  --nlist 32768 --nprobe 64 --M 32

# ---------------------------------------------------------------------
# 5) Build Arrow passage dataset  ➜ wiki_passages_dataset_hnsw/
# ---------------------------------------------------------------------
python "$SRC/wiki_build_passages.py" \
  --shard_emb_glob  "$REPO/data/wiki_emb_shard*.npy" \
  --shard_meta_glob "$REPO/data/wiki_emb_meta_shard*.npy" \
  --out_dir   "wiki_passages_dataset_hnsw" \
  --index_file "wiki_hnsw.faiss"

# ---------------------------------------------------------------------
# 6) Offline DPR retrieval  ➜ pretrieved_triviaqa_hnsw.jsonl
# ---------------------------------------------------------------------
python "$SRC/wiki_prep_retrieval.py" \
  --data_dir "$REPO/data_dir" \
  --index_file    "$DATA/wiki_hnsw.faiss" \
  --meta_file     "$DATA/wiki_hnsw_meta.npy" \
  --k 15\
  --split 0.8 \
  --max_samples 2000\
  --output_json pretrieved_triviaqa_hnsw.jsonl 

# ---------------------------------------------------------------------
# 7.1) Generator-only fine-tuning (T5-LoRA)
# ---------------------------------------------------------------------
python "$SRC/train_generator.py" \
  --pretrieved_json "pretrieved_triviaqa_hnsw.jsonl" \
  --output_dir      "t5_lora_ft" \
  --epochs 3 --batch_size 4 --learning_rate 1e-4 \
  --val_ratio 0.1 --max_samples 20000

# ---------------------------------------------------------------------
# 7.2) Clone the End-to-end RAG from Github and fine-tuning (retriever + generator)
# ---------------------------------------------------------------------

RAG="$myscratch/rag-end2end-retriever" && \
git clone --depth 1 https://github.com/huggingface/transformers-research-projects.git "$myscratch/transformers-rp" && \
git -C "$myscratch/transformers-rp" checkout 362a490dc36e91359fe76a7a707dc29e663196b2 && \
ln -s "$myscratch/transformers-rp/rag-end2end-retriever" "$RAG"

python "$RAG/finetune_rag.py" \
  --data_dir  "$REPO/data_dir" \
  --output_dir "rag_end2end_ft" \
  --model_name_or_path facebook/rag-token-base \
  --model_type rag_token \
  --fp16 --gpus 1 --profile \
  --do_train --end2end --do_predict \
  --train_batch_size 16 --eval_batch_size 1 \
  --max_source_length 128 --max_target_length 25 \
  --val_max_target_length 25 --test_max_target_length 25 \
  --label_smoothing 0.1 --dropout 0.1 --attention_dropout 0.1 \
  --weight_decay 0.001 --adam_epsilon 1e-08 \
  --max_grad_norm 0.1 --lr_scheduler polynomial \
  --learning_rate 3e-05 --num_train_epochs 3 \
  --warmup_steps 100 \
  --distributed_retriever ray --num_retrieval_workers 4 \
  --gradient_accumulation_steps 8 \
  --passages_path "wiki_passages_dataset_hnsw" \
  --index_path    "wiki_passages_dataset_hnsw/embeddings.faiss" \
  --index_name custom \
  --context_encoder_name facebook/dpr-ctx_encoder-multiset-base \
  --indexing_freq 1000

# ---------------------------------------------------------------------
# 8) Evaluate all six modes
# ---------------------------------------------------------------------
python "$SRC/evaluation.py" \
  --index_file "wiki_hnsw.faiss" \
  --meta_file  "wiki_hnsw_meta.npy" \
  --k 15 --sample_size 12000 --mode all \
  --lora_dir t5_lora_ft \
  --ckpt_dir rag_end2end_ft/checkpoint-best \
  --passages_path wiki_passages_dataset_hnsw \
  --index_path    wiki_passages_dataset_hnsw/embeddings.faiss \
  --inspect 10

echo "done"
