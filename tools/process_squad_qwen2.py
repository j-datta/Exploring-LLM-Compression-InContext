import multiprocessing
import os
import time
import torch
import json
import sys
import random
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from arguments import get_args
    
class Encoder(object):
    def __init__(self, args):
        self.args = args
        
    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)

    def encode(self, line):
        line = json.loads(line)
        
        # Extract necessary fields from the MLQA dataset
        id = line.get("id", "")
        prompt=line.get("prompt", "")
        output=line.get("output", "")
        #question = line.get("question", "")
        #context = line.get("context", "")
        #answer = line.get("answer", "")
        #answers = line.get("answers", {}).get("text", [])

        # Use the first answer as the target response or provide a default message
        #answer = answers[0] if answers else "No answer available."

        # Instruction template for MLQA dataset
        #if "input" not in line or len(line["input"]) == 0:
         #   if self.args.model_type != "qwen":
          #      template = (
           #         "Please answer the following question based on the given context. "
            #        "Provide ONLY the answer to the following question based on the given context. "
             #       "Do NOT include any explanations or additional information.\n\n"
              #      "### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:\n"
               # )
            #else:
             #   template = (
              #      "<|im_start|>user\nPlease answer the following question based on the given context. "
               #     "Provide ONLY the answer to the following question based on the given context. "
                #    "Do NOT include any explanations or additional information.\n\n"
                 #   "### Context:\n{context}\n\n### Question:\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
                #)
                #template = (
                 #   "<|im_start|>user\nBitte beantworten Sie die folgende Frage basierend auf dem gegebenen Kontext. Geben Sie NUR die Antwort auf die folgende Frage basierend auf dem gegebenen Kontext. Fügen Sie KEINE Erklärungen oder zusätzlichen Informationen hinzu.\n\n"
                  #  "### Kontext:\n{context}\n\n### Frage:\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
                   # )
        #prompt = f"<|im_start|>Context: {context}\n\nQuestion: {question}\n\nAnswer:<|im_end|>"
                
        #prompt = template.format(context=context, question=question)

        # Tokenize prompt and response
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = Encoder.tokenizer.encode(prompt + output, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]        
        response_tokens = full_tokens[len(prompt_tokens):]

        # Truncate the prompt if it exceeds the maximum length
        if len(prompt_tokens) > self.args.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.args.max_prompt_length]
        
        return line, id, prompt, prompt_tokens, response_tokens, len(line), output


def main():
    args = get_args()

    # Create directories if they don't exist
    if 'generated' not in args.processed_data_dir:
        args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)

    # Load train and validation data
    with open(os.path.join(args.data_dir, "train_seed_42.jsonl")) as f:
        train_data = f.readlines()
    
    train_split_ratio = args.train_split_ratio
    random.shuffle(train_data)
    
    train_size = int(len(train_data) * train_split_ratio)
    new_train_data = train_data[:train_size] 
    new_validation_data = train_data[train_size:]

    all_data = {
        "train": new_train_data,
        "valid": new_validation_data
    }
    
    # Process each data split
    for split in all_data:
        encoder = Encoder(args)
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        #encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=10)
        proc_start = time.time()
        total_bytes_processed = 0
        
        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

       # if args.model_type != "qwen":
        #    binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
        #else:
         #   binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint32)
         
        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint32)


        inst_num = 0
        print("#" * 10, split, "#" * 10)
        
        prompt_lens = []
        response_lens = []
        
        with open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w", encoding='utf-8') as json_file:
            for lid, (line, id, prompt_str, prompt, response, bytes_processed, answer) in enumerate(encoded_docs):
                total_bytes_processed += bytes_processed
                if prompt is None:
                    continue
                
                if args.only_prompt:
                    if len(prompt) < args.max_length:
                        binary_builder.add_item(torch.IntTensor(prompt))
                    else:
                        continue
                else:
                    binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))
                    
                # Write to the JSONL file with instruction, prompt, and output
                json_file.write(json.dumps({
                    'id': id,
                    "prompt": prompt_str,
                    "output": answer
                    }, ensure_ascii=False) + "\n")

                prompt_lens.append(len(prompt))
                response_lens.append(len(response))

                inst_num += 1
                if lid % 1000 == 0:
                    current = time.time()
                    elapsed = current - proc_start
                    mbs = total_bytes_processed / elapsed / 1024 / 1024
                    print(f"Processed {lid} documents. {inst_num} instances.",
                          f"({lid / elapsed} docs/s, {mbs} MB/s).",
                          file=sys.stderr)

        binary_builder.finalize(idx_file)
        pool.close()

        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))


if __name__ == '__main__':
    main()