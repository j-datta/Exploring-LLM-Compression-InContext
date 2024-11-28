import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load the pre-trained model and tokenizer
model_name = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Handle EOS and PAD tokens
EOS_TOKEN = tokenizer.eos_token if tokenizer.eos_token else '<|im_end|>'
eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the MLQA dataset (English split) from JSONL files
train_path = "/home/IAIS/jdatta/distillm-new/processed_data_test/mlqa_en/full/qwen/train.jsonl"
validation_path = "/home/IAIS/jdatta/distillm-new/processed_data_test/mlqa_en/full/qwen/valid.jsonl"
train_dataset = load_dataset("json", data_files={"train": train_path})["train"]
validation_dataset = load_dataset("json", data_files={"validation": validation_path})["validation"]

def preprocess_function(examples):
    prompts = examples["prompt"]
    answers = examples["output"]

    # Concatenate the prompt and answer with EOS tokens
    inputs = [prompt + EOS_TOKEN + answer + EOS_TOKEN for prompt, answer in zip(prompts, answers)]

    tokenized_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = tokenized_inputs["input_ids"]

    labels = []
    for i, input_id in enumerate(input_ids):
        input_id_list = input_id.tolist()
        try:
            prompt_end_index = input_id_list.index(eos_token_id) + 1
        except ValueError:
            prompt_end_index = len(input_id_list)

        # Initialize label with all values as -100
        label = [-100] * len(input_id_list)

        # Set labels for the answer tokens (excluding padding tokens)
        answer_indices = range(prompt_end_index, len(input_id_list))
        for idx in answer_indices:
            if input_id_list[idx] != tokenizer.pad_token_id:
                label[idx] = input_id_list[idx]

        labels.append(label)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_validation = validation_dataset.map(preprocess_function, batched=True)

# Prepare the LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    optim="adamw_torch",
    logging_dir="./logs",
    logging_steps=50,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    max_steps=-1,
    save_strategy="epoch",
    save_total_limit=10,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
)

class SaveModelAndTokenizerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        tokenizer = kwargs['tokenizer']

        output_dir = os.path.join(args.output_dir, f"epoch_{int(state.epoch)}")
        os.makedirs(output_dir, exist_ok=True)

        # Save the adapter weights
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"Model and tokenizer saved to {output_dir} at epoch {int(state.epoch)}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    callbacks=[SaveModelAndTokenizerCallback()],
)

trainer.train()