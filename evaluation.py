#!/usr/bin/env python
import argparse, json, gc, time
import faiss, numpy as np, torch
from datasets import load_dataset
from transformers import (
    RagTokenizer, RagTokenForGeneration, RagRetriever,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast,
)
from peft import PeftModel

MODES = ["base", "lora", "rerank", "iter", "gen_only", "rag"]

def normalize(text):
    return " ".join(text.lower().split())

def exact_match(pred, golds):
    return int(any(normalize(pred)==normalize(g) for g in golds))

def token_f1(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    com = set(pt)&set(gt)
    if not com: return 0.
    prec, rec = len(com)/len(pt), len(com)/len(gt)
    return 2*prec*rec/(prec+rec)

def build_t5(name, lora, device):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
    if lora: mdl = PeftModel.from_pretrained(mdl, lora)
    return tok, mdl.to(device).eval()

def t5_answer(tok, mdl, q, ctxs, dev, maxl):
    prompt = "answer_question: "+q + ("" if not ctxs else
        " context: "+" ".join(f"[{i+1}] {c}" for i,c in enumerate(ctxs)))
    inp = tok(prompt, return_tensors="pt", truncation=True,
              padding="max_length", max_length=512).to(dev)
    out = mdl.generate(**inp, max_length=maxl, num_beams=5, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)

def batch_rerank(tok,mdl,q,ctxs,dev,nr,bz=8):
    prompts = [f"Query: {q}\nContext: {c}\nScore:" for c in ctxs]
    scores=[]
    for i in range(0,len(prompts),bz):
        b=prompts[i:i+bz]
        inp=tok(b,return_tensors="pt",truncation=True,
                padding="max_length",max_length=256).to(dev)
        ids=mdl.generate(**inp,max_length=8,num_beams=4)
        dec=tok.batch_decode(ids,skip_special_tokens=True)
        for s in dec:
            try: scores.append(float(s.strip()))
            except: scores.append(0.)
    ranked=[c for _,c in sorted(zip(scores,ctxs),reverse=True)]
    return ranked[:nr]

def build_rag(ckpt,ppath,ipath,dev):
    tok=RagTokenizer.from_pretrained(ckpt)
    mdl=RagTokenForGeneration.from_pretrained(ckpt)
    retr=RagRetriever.from_pretrained(
        "facebook/rag-token-base", index_name="custom",
        passages_path=ppath, index_path=ipath)
    mdl.set_retriever(retr)
    return tok, mdl.to(dev).eval()

def rag_answer(tok,mdl,q,dev,maxl):
    inp=tok(q,return_tensors="pt",truncation=True,
            padding="max_length",max_length=512).to(dev)
    out=mdl.generate(**inp,min_length=1,max_length=maxl,
                     num_beams=1,do_deduplication=False,
                     use_cache=True)
    return tok.batch_decode(out,skip_special_tokens=True)[0]

def evaluate(mode, args, device, ds):
    # prepare
    if mode in MODES[:-1]:
        qtok = DPRQuestionEncoderTokenizerFast.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base")
        qenc = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        ).to(device).eval()
        t5_tok, t5_mdl = build_t5("t5-base",
                                  args.lora_dir if mode=="lora" else None,
                                  device)
        faiss_idx = faiss.read_index("wiki_passages_dataset_hnsw/embeddings.faiss")
        faiss_idx.nprobe = args.k
        passages = np.load("wiki_hnsw_meta.npy",allow_pickle=True)
    if mode=="rag":
        rag_tok, rag_mdl = build_rag(args.ckpt_dir,args.passages_path,
                                     args.index_path,device)

    # run & time
    torch.cuda.empty_cache()
    if device.type=="cuda": torch.cuda.synchronize()
    t0 = time.time()

    ems, f1s, rec = [], [], 0
    for ex in ds:
        q = ex["question"]
        golds = ex["answer"]["aliases"] or [ex["answer"]["value"]]
        if mode=="gen_only":
            pred = t5_answer(t5_tok,t5_mdl,q,[],device,args.max_len)
        elif mode in MODES[:-1]:
            with torch.no_grad():
                emb = qenc(**qtok(q,return_tensors="pt").to(device)).pooler_output.cpu().numpy()
            _, ids = faiss_idx.search(emb, args.k)
            ctxs = [passages[i] for i in ids[0]]
            if mode=="rerank":
                ctxs = batch_rerank(t5_tok,t5_mdl,q,ctxs,device,args.n_rerank)
            if mode=="iter":
                cur=q
                for _ in range(args.iterations):
                    ans = t5_answer(t5_tok,t5_mdl,cur,ctxs,device,args.max_len)
                    cur += " "+ans
                    with torch.no_grad():
                        emb = qenc(**qtok(cur,return_tensors="pt").to(device)).pooler_output.cpu().numpy()
                    _, ids = faiss_idx.search(emb,args.k)
                    ctxs = [passages[i] for i in ids[0]]
                pred = ans
            if mode in ["base","lora","rerank"]:
                pred = t5_answer(t5_tok,t5_mdl,q,ctxs,device,args.max_len)
            rec += int(any(normalize(g) in c.lower() for g in golds for c in ctxs))
        else:  # rag
            pred = rag_answer(rag_tok,rag_mdl,q,device,args.max_len)

        ems.append(exact_match(pred,golds))
        f1s.append(max(token_f1(pred,g) for g in golds))
        gc.collect()

    if device.type=="cuda": torch.cuda.synchronize()
    elapsed = time.time()-t0
    qps = len(ds)/elapsed

    # save stats
    with open(f"throughput_stats_{mode}.json","w") as f:
        json.dump({"mode":mode,
                   "answered":len(ds),
                   "seconds":elapsed,
                   "qps":qps},f,indent=2)
    res = {"EM":float(np.mean(ems)),
           "F1":float(np.mean(f1s)),
           f"R@{args.k}":rec/len(ds)}
    with open(f"eval_results_{mode}.json","w") as f:
        json.dump({mode:res},f,indent=2)
    print(f"{mode}: {len(ds)} queries in {elapsed:.1f}s → {qps:.1f} q/s | EM={res['EM']:.3f}, F1={res['F1']:.3f}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=MODES+["all"],required=True)
    p.add_argument("--sample_size",type=int,default=200)
    p.add_argument("--k",type=int,default=15)
    p.add_argument("--n_rerank",type=int,default=6)
    p.add_argument("--iterations",type=int,default=2)
    p.add_argument("--max_len",type=int,default=25)
    p.add_argument("--lora_dir")
    p.add_argument("--ckpt_dir")
    p.add_argument("--passages_path")
    p.add_argument("--index_path")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset("trivia_qa","rc.nocontext",split="validation") \
           .select(range(args.sample_size))

    modes = MODES if args.mode=="all" else [args.mode]
    for m in modes:
        evaluate(m, args, device, ds)

if __name__=="__main__":
    main()
