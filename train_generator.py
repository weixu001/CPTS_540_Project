#!/usr/bin/env python
"""
LoRA-fine-tune T5-base on (question, top-k-contexts → answer) JSONL.
Writes only the adapter (≈15 MB).

Example:
  python train_t5_lora.py \
    --pretrieved_json  pretrieved_triviaqa.jsonl \
    --output_dir       t5_lora_ft \
    --epochs           3 \
    --batch_size       4 \
    --learning_rate    1e-4  \
    --val_ratio        0.1    \
    --max_samples      20000
"""
import accelerate
from accelerate import Accelerator as _OldAccelerator
class Accelerator(_OldAccelerator):
    def __init__(self, *args, **kwargs):
        for bad in ("dispatch_batches", "use_seedable_sampler"):
            kwargs.pop(bad, None)
        super().__init__(*args, **kwargs)

import accelerate
accelerate.Accelerator = Accelerator

import argparse, json, random
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np


def read_jsonl(path, max_samples=None):
    rows = []
    with open(path) as f:
        for i, ln in enumerate(f):
            if max_samples and i >= max_samples:
                break
            rows.append(json.loads(ln))
    return rows

def build_ds(rows):
    return Dataset.from_list([{
        "src": f"{ex['question']} " + " ".join(ex["contexts"]),
        "tgt": ex["answer"]
    } for ex in rows])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrieved_json", required=True)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--learning_rate",   type=float, default=1e-4)
    p.add_argument("--val_ratio",       type=float, default=0.1)
    p.add_argument("--max_samples",     type=int,   default=None)
    args = p.parse_args()

    # 1) read & split
    rows = read_jsonl(args.pretrieved_json, args.max_samples)
    random.shuffle(rows)
    split = int(len(rows)*(1-args.val_ratio))
    ds = DatasetDict({
        "train":      build_ds(rows[:split]),
        "validation": build_ds(rows[split:]),
    })

    # 2) tokenizer & base T5
    tok   = AutoTokenizer.from_pretrained("t5-base")
    base  = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    # 3) wrap in LoRA
    peft_cfg = LoraConfig(
        task_type      = TaskType.SEQ_2_SEQ_LM,
        r              = 16,
        lora_alpha     = 32,
        lora_dropout   = 0.05,
        target_modules = ["q","v"],   # T5 uses "q", "v" for attention
    )
    model = get_peft_model(base, peft_cfg)

    # 4) preprocess
    def preprocess(batch):
        x = tok(batch["src"],  max_length=512, truncation=True, padding="max_length")
        y = tok(batch["tgt"],  max_length= 64, truncation=True, padding="max_length")
        return {
            "input_ids":      x["input_ids"],
            "attention_mask": x["attention_mask"],
            "labels":         y["input_ids"],
        }
    ds = ds.map(preprocess, batched=True, remove_columns=["src","tgt"])
    ds.set_format("torch")

    # 5) collator & metrics
    collator = DataCollatorForSeq2Seq(tok, model=model)
    rouge    = load_metric("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        pad_id = tok.pad_token_id

        preds  = np.where(preds  == -100, pad_id, preds)
        labels = np.where(labels == -100, pad_id, labels)
        decoded_preds  = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        res = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {"rougeL_f1": res["rougeL"]}

    # 6) training args
    targs = Seq2SeqTrainingArguments(
        output_dir               = args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size =args.batch_size,
        learning_rate            = args.learning_rate,
        num_train_epochs         = args.epochs,
        logging_steps            = 100,
        evaluation_strategy      = "steps",
        eval_steps              = 200,
        save_strategy            = "steps",
        save_steps               = 200,
        save_total_limit         = 2,
        load_best_model_at_end   = True,
        metric_for_best_model    = "rougeL_f1",
        predict_with_generate    = True,
        generation_max_length    =  64,
        fp16                     = torch.cuda.is_available(),
        report_to                = "none",
    )

    # 7) trainer
    trainer = Seq2SeqTrainer(
        model           = model,
        args            = targs,
        train_dataset   = ds["train"],
        eval_dataset    = ds["validation"],
        tokenizer       = tok,
        data_collator   = collator,
        compute_metrics = compute_metrics,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("✅ LoRA adapter saved to", args.output_dir)


if __name__=="__main__":
    main()
