"""Export a trained LoRA adapter to a 4-bit NF4 inference package.

This script only runs after M5 has produced an adapter. It never claims a
quantized model exists until the export and checksum files are written.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base-model",required=True); ap.add_argument("--adapter",required=True); ap.add_argument("--output",required=True); args=ap.parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError as exc: raise SystemExit("Install transformers peft bitsandbytes torch") from exc
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(args.base_model,quantization_config=config,device_map="auto",torch_dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(model,args.adapter); model.save_pretrained(out,safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(out)
    (out/"QUANTIZATION_MANIFEST.json").write_text(json.dumps({"base_model":args.base_model,"adapter":args.adapter,"quantization":"4bit_nf4","compute_dtype":"bfloat16","status":"exported"},indent=2),encoding="utf-8")
if __name__ == "__main__": main()
