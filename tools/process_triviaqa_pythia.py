import multiprocessing
import os
import time
import torch
import json
import sys
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from arguments import get_args

class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        line = json.loads(line)
        
        # Extract necessary fields for TriviaQA
        id = line.get("question_id", "")
        question = line.get("question", "")
        #context = line.get("context", "")
        #answers = line.get("answer", {}).get("aliases", [])
        answer = line.get("answer", "No answer available.")

        # Use the first answer as the correct answer, or a placeholder if none available
        #answer = answers[0] if answers else "No answer available."
        
        # Create the prompt
        #prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        prompt = f"Question: {question}\n\nAnswer:"

        # Tokenization
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = Encoder.tokenizer.encode(f" {answer}", add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]

        if len(prompt_tokens) > self.args.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.args.max_prompt_length]

        return line, id, prompt, prompt_tokens, response_tokens, len(line), answer


def main():
    print("OK")
    args = get_args()
        
    args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)
    os.makedirs(args.processed_data_dir, exist_ok=True)

    # Load each split separately
    all_data_files = {
        "train": "train.jsonl",
        "valid": "validation.jsonl"
    }
    
    all_data = {}
    for split, file_name in all_data_files.items():
        with open(os.path.join(args.data_dir, file_name)) as f:
            all_data[split] = f.readlines()

    # Process each split without additional shuffling or splitting
    for split in all_data:
        encoder = Encoder(args)
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0
        
        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
        
        inst_num = 0
        print("#"*10, split, "#"*10)
        
        prompt_lens = []
        response_lens = []
        
        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")
        
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

            # Write to the JSONL file with id, prompt, and output keys
            json_file.write(json.dumps({
                "id": id,              
                "prompt": prompt_str,  
                "output": answer      
            }) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        binary_builder.finalize(idx_file)
        pool.close()
        json_file.close()
                
        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))


if __name__ == '__main__':
    main()
