"""
train_qwen_qlora_t4.py

T4-safe QLoRA training entrypoint for Kern chat datasets.

Expected dataset format (JSONL):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA training for Qwen on Colab T4.")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B", help="Base HF model.")
    parser.add_argument("--train-file", required=True, help="Train JSONL path.")
    parser.add_argument("--valid-file", required=True, help="Validation JSONL path.")
    parser.add_argument("--output-dir", default="outputs/qwen-kern-qlora-t4", help="Output dir.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--max-valid-samples", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank. Raise to 32 for stronger adapters if memory allows.",
    )
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="'auto', 'none', or explicit checkpoint path.",
    )
    return parser.parse_args()


def to_chat_text(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each row must contain a non-empty 'messages' list.")

    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        chunks: list[str] = []
        for m in messages:
            role = str(m.get("role", "user")).strip()
            content = str(m.get("content", ""))
            chunks.append(f"<|{role}|>\n{content}\n")
        text = "".join(chunks)
    return {"text": text}


def resolve_resume_checkpoint(output_dir: Path, resume_arg: str) -> str | None:
    if resume_arg.lower() == "none":
        return None
    if resume_arg.lower() != "auto":
        return resume_arg

    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This script is intended for Colab GPU runtime.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except ValueError as exc:
        msg = str(exc)
        if "model type `qwen3_5`" in msg and "does not recognize this architecture" in msg:
            raise RuntimeError(
                "Your transformers version does not support Qwen3.5 yet. "
                "Upgrade in Colab with:\n"
                "  pip install -U git+https://github.com/huggingface/transformers.git\n"
                "Then restart runtime and run again."
            ) from exc
        raise
    model.config.use_cache = False

    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.valid_file},
    )
    train_ds = ds["train"]
    valid_ds = ds["validation"]

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_valid_samples > 0:
        valid_ds = valid_ds.select(range(min(args.max_valid_samples, len(valid_ds))))

    train_ds = train_ds.map(
        lambda x: to_chat_text(x, tokenizer),
        remove_columns=train_ds.column_names,
        desc="Formatting train chat template",
    )
    valid_ds = valid_ds.map(
        lambda x: to_chat_text(x, tokenizer),
        remove_columns=valid_ds.column_names,
        desc="Formatting valid chat template",
    )

    lora_targets = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )

    train_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text",
        packing=False,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        peft_config=lora_config,
    )

    resume_ckpt = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        print("Starting fresh training run.")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metadata = {
        "model_name": args.model_name,
        "train_file": args.train_file,
        "valid_file": args.valid_file,
        "output_dir": str(output_dir),
        "final_dir": str(final_dir),
        "num_train_rows": len(train_ds),
        "num_valid_rows": len(valid_ds),
        "max_seq_length": args.max_seq_length,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "resume_from_checkpoint": resume_ckpt,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
