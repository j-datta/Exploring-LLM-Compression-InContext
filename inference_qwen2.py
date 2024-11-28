import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the fine-tuned model and tokenizer
adapter_model_path = "/home/IAIS/jdatta/distillm-new/results/qwen2/train/minillm_init/Qwen2-0.5B_lora"
model = PeftModel.from_pretrained(model, adapter_model_path)
model.eval()  

# Set pad_token_id if it's not defined in the tokenizer (use eos_token_id as default)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Function to read the input jsonl file
def load_jsonl_file(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load the validation data
validation_file = "/home/IAIS/jdatta/distillm-new/data/mlqa/english/valid.jsonl"
input_data = load_jsonl_file(validation_file)

def inference_and_save_answers_to_jsonl(input_data, output_file):
    results = []

    # Iterate over the input data (each entry in the validation file)
    for item in input_data:
        prompt = item["prompt"]
        
        # Modify the prompt structure to ensure the model expects an answer
        modified_prompt = f"{prompt}\nAnswer:"
        inputs = tokenizer(modified_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Generate the output using the model with sampling and generation options
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  
                max_new_tokens=100,            
                do_sample=True,                 
                top_k=0,                       
                top_p=1.0,                      
                temperature=0.8,
                early_stopping=False,               
                pad_token_id=tokenizer.pad_token_id,  
                eos_token_id=tokenizer.eos_token_id
            )

        # After generation
        generated_tokens = outputs[0][input_ids.shape[-1]:]
        generated_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Generated Answer: {generated_answer}")

        results.append({"text": generated_answer.strip()})

    # Write all results to the output file after processing all inputs
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Inference completed. Results saved to {output_file}")

output_file = "/home/IAIS/jdatta/distillm-new/results/qwen2/validation_answers.jsonl"

inference_and_save_answers_to_jsonl(input_data, output_file)