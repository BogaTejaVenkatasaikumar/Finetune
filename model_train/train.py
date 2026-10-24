
import json
import inspect
from pathlib import Path

import torch
import trl
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import SFTTrainer

# Try to support both newer and older TRL APIs
try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_DIR / "data" / "splits" / "train.jsonl"
VALIDATION_FILE = PROJECT_DIR / "data" / "splits" / "validation.jsonl"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "bro-slm-lora"
FINAL_ADAPTER_DIR = OUTPUT_DIR / "final_adapter"

MAX_SEQ_LENGTH = 512
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
SEED = 42


# ==========================================
# 2. MAIN TRAINING FUNCTION
# ==========================================

def main():
    set_seed(SEED)

    print("=" * 55)
    print("QWEN SLM FINE-TUNING")
    print("=" * 55)
    print("TRL version:", trl.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------
    # Check dataset paths
    # --------------------------------------

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------
    # Load datasets
    # --------------------------------------

    print("\nLoading datasets...")

    datasets = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VALIDATION_FILE),
        },
    )

    train_dataset = datasets["train"]
    validation_dataset = datasets["validation"]

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    if "messages" not in train_dataset.column_names:
        raise ValueError(
            "Expected a 'messages' field in each JSONL example."
        )

    # Basic conversation-format validation
    for index, example in enumerate(train_dataset):
        messages = example.get("messages")

        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"Invalid messages list at training example {index}"
            )

        roles = {
            message.get("role")
            for message in messages
            if isinstance(message, dict)
        }

        if "user" not in roles or "assistant" not in roles:
            raise ValueError(
                f"Example {index} needs user and assistant messages."
            )

    # --------------------------------------
    # Load tokenizer
    # --------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------
    # Load pretrained model
    # --------------------------------------

    print("Loading pretrained model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
    )

    model.config.use_cache = False

    # --------------------------------------
    # Configure LoRA
    # --------------------------------------

    print("Configuring LoRA...")

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # --------------------------------------
    # Configure training
    # --------------------------------------

    print("Configuring training arguments...")

    trainer_signature = inspect.signature(SFTTrainer.__init__)
    trainer_params = trainer_signature.parameters

    # Select the argument name supported by this TRL version
    if "processing_class" in trainer_params:
        tokenizer_argument = "processing_class"
    elif "tokenizer" in trainer_params:
        tokenizer_argument = "tokenizer"
    else:
        raise RuntimeError(
            "Could not identify the tokenizer argument supported "
            "by this SFTTrainer version. Run the signature diagnostic "
            "and check your installed TRL documentation."
        )

    # Find the sequence-length parameter supported by this version
    if SFTConfig is not None:
        config_signature = inspect.signature(SFTConfig.__init__)
        config_params = config_signature.parameters

        if "max_length" in config_params:
            sequence_length_argument = "max_length"
        elif "max_seq_length" in config_params:
            sequence_length_argument = "max_seq_length"
        else:
            sequence_length_argument = None
    else:
        sequence_length_argument = None

    # Most recent TRL versions use SFTConfig.
    # Older versions can use TrainingArguments.
    from transformers import TrainingArguments

    common_args = dict(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
    )

    if SFTConfig is not None and sequence_length_argument:
        config_kwargs = dict(common_args)
        config_kwargs[sequence_length_argument] = MAX_SEQ_LENGTH
        training_args = SFTConfig(**config_kwargs)
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": validation_dataset,
            "peft_config": peft_config,
            tokenizer_argument: tokenizer,
        }
    else:
        training_args = TrainingArguments(**common_args)
        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": validation_dataset,
            "peft_config": peft_config,
            tokenizer_argument: tokenizer,
        }

        if "max_seq_length" in trainer_params:
            trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
        elif "max_length" in trainer_params:
            trainer_kwargs["max_length"] = MAX_SEQ_LENGTH
        else:
            raise RuntimeError(
                "This TRL version does not expose a recognized "
                "sequence-length argument. Please share the "
                "printed SFTTrainer signature."
            )

    # --------------------------------------
    # Create trainer
    # --------------------------------------

    print("Creating trainer...")

    trainer = SFTTrainer(**trainer_kwargs)

    # --------------------------------------
    # Train
    # --------------------------------------

    print("\nStarting training...")
    print("Sequence length:", MAX_SEQ_LENGTH)
    print("Epochs:", NUM_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS,
    )

    train_result = trainer.train()

    # --------------------------------------
    # Evaluate on validation data
    # --------------------------------------

    print("\nEvaluating on validation data...")

    validation_metrics = trainer.evaluate(
        metric_key_prefix="validation"
    )

    print("Validation metrics:", validation_metrics)

    # --------------------------------------
    # Save adapter and tokenizer
    # --------------------------------------

    print("\nSaving trained adapter...")

    FINAL_ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))

    metrics = {
        "train": train_result.metrics,
        "validation": validation_metrics,
        "model_name": MODEL_NAME,
        "epochs": NUM_EPOCHS,
        "max_seq_length": MAX_SEQ_LENGTH,
    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2, default=str)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)
    print("Adapter directory:", FINAL_ADAPTER_DIR)
    print("Metrics file:", OUTPUT_DIR / "metrics.json")
    print("You can now load the adapter for inference.")


if __name__ == "__main__":
    main()
