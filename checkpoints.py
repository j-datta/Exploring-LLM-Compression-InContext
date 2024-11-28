#import sys
#sys.path.append('/path/to/transformers/src')

#from transformers import AutoTokenizer, AutoModelForCausalLM
#from transformers.models.mistral import MistralForCausalLM, MistralConfig

#config = MistralConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
#tokenizer =AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
#model = model.half()

# = MistralConfig.from_pretrained("micro_mistral")
#tokenizer =AutoTokenizer.from_pretrained("micro_mistral")
#model = MistralForCausalLM.from_pretrained("micro_mistral")
#model = model.half()

#from transformers import AutoTokenizer, AutoModelForCausalLM

#model = "Qwen/Qwen2-0.5B"
#tokenizer = AutoTokenizer.from_pretrained(model)
#model = AutoModelForCausalLM.from_pretrained(model)

# Save the tokenizer and model
#save_directory = "/home/IAIS/jdatta/distillm-new/checkpoints/Qwen2-0.5B"
#tokenizer.save_pretrained(save_directory)
#model.save_pretrained(save_directory)

from transformers import AutoModelForCausalLM, AutoTokenizer

model = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained(model)

target_vocab_size = 151936
model.resize_token_embeddings(target_vocab_size)

# Save the tokenizer and model
save_directory = "/home/IAIS/jdatta/distillm-new/checkpoints/Qwen2.5-7B"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

#from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

#from transformers.models.mistral.tokenization_mistral import MistralTokenizer
#from transformers.models.mistral.configuration_mistral import MistralConfig

#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", config=config)
#model=model.half()

#save_directory = "/home/IAIS/jdatta/distillm/checkpoints/checkpointsMistral-7B-v0.1"

#tokenizer.save_pretrained(save_directory)
#model.save_pretrained(save_directory)

#from transformers import GPT2Tokenizer, GPT2LMHeadModel

#tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-xl")
#model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-xl")
#model=model.half()

#save_directory = "/home/IAIS/jdatta/distillm/checkpoints/gpt2-xlarge"

#tokenizer.save_pretrained(save_directory)
#model.save_pretrained(save_directory)
