
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "outputs/bro-slm-lora/final_adapter"

MAX_NEW_TOKENS = 200


# ==================================================
# 2. SELECT DEVICE
# ==================================================

if torch.cuda.is_available():
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("GPU not available. Using CPU.")


# ==================================================
# 3. LOAD TOKENIZER
# ==================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==================================================
# 4. LOAD BASE MODEL
# ==================================================

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=(
        torch.float16
        if device == "cuda"
        else torch.float32
    ),
    low_cpu_mem_usage=True,
)


# ==================================================
# 5. LOAD TRAINED LORA ADAPTER
# ==================================================

print("Loading trained LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

model = model.to(device)
model.eval()

print("\nYour fine-tuned model is ready!")
print("Type 'exit' or 'quit' to stop.\n")


# ==================================================
# 6. INTERACTIVE INFERENCE LOOP
# ==================================================

while True:

    try:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Closing model session.")
            break

        if not user_input:
            continue

        # ------------------------------------------
        # 7. FORMAT THE USER MESSAGE
        # ------------------------------------------

        messages = [
            {
                "role": "user",
                "content": user_input,
            }
        ]

        # Request dictionary output containing input_ids
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move every input tensor to the selected device
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ------------------------------------------
        # 8. GENERATE RESPONSE
        # ------------------------------------------

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # ------------------------------------------
        # 9. DECODE ONLY NEW TOKENS
        # ------------------------------------------

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[0][input_length:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        print(f"\nModel: {response}\n")

    except KeyboardInterrupt:
        print("\nSession stopped.")
        break

    except Exception as error:
        print(f"\nError: {type(error).__name__}: {error}\n")
