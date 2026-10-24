# Bro SLM Fine-Tuning

A small end-to-end project for cleaning a chat dataset, fine-tuning
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) with
LoRA, and trying the resulting adapter interactively.

> This is an experimental fine-tune, not a general-purpose or production-ready
> model. It was trained for one epoch on a small, 108-example dataset.

## What is included

- `clean_data/` — dataset cleaning, preparation, and train/validation/test
  splitting scripts.
- `data/` — source and cleaned conversations, split JSONL files, and a
  cleaning report.
- `model_train/train.py` — supervised fine-tuning with TRL and PEFT LoRA.
- `model_train/infer.py` — interactive command-line inference.
- `outputs/bro-slm-lora/final_adapter/` — trained LoRA adapter and tokenizer.
- `outputs/bro-slm-lora/checkpoint-11/` — training checkpoint.
- `outputs/bro-slm-lora/metrics.json` — recorded training and validation
  metrics.

The adapter contains only the LoRA weights; it needs the base model to run.

## Requirements

- Python 3.10 or newer
- Internet access on first run to download the base model
- PyTorch and the Hugging Face packages below
- A CUDA-capable GPU is recommended for training; inference can also run on CPU

From the project root, create and activate a virtual environment, then install
the packages:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch transformers peft trl datasets accelerate
```

Install a PyTorch build appropriate for your hardware if the default `torch`
installation does not support your GPU.

## Run inference

From the project root:

```powershell
python model_train\infer.py
```

Enter a prompt at `You:`. Type `exit` or `quit` to close the session. The
script loads the base model and the adapter from
`outputs/bro-slm-lora/final_adapter`.

## Re-train

The training script expects `data/splits/train.jsonl` and
`data/splits/validation.jsonl` in the format used by the included dataset.
From the project root, run:

```powershell
python model_train\train.py
```

Training writes checkpoints and metrics to `outputs/bro-slm-lora/`, and saves
the final adapter to `outputs/bro-slm-lora/final_adapter/`. Training again
will update files in that output directory.

To rebuild the data from the source dataset, run these scripts from the
project root, in order:

```powershell
python clean_data\clean_data.py
python clean_data\prepared-dataset.py
```

The split script groups records with matching normalized user questions to
reduce duplicate-question leakage across splits.

## Training snapshot

The included run fine-tuned the base model for one epoch with a maximum
sequence length of 512. Its recorded validation loss is approximately 2.49.
This metric alone does not establish response quality; test the model on
representative prompts before relying on its outputs.

## Notes

- The adapter was trained from `Qwen/Qwen2.5-0.5B-Instruct`; review that base
  model's license and usage terms before redistributing or deploying it.
- Generated responses can be incorrect or incomplete. Validate important
  information independently.
- The included dataset and metrics describe this particular small experiment,
  not a comprehensive evaluation.
