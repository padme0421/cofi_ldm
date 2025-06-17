from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import torch
import logging
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
import os
import huggingface_hub
from dotenv import load_dotenv
from torch.nn.utils.rnn import pad_sequence
import argparse
import json
from get_embeddings import get_prompt

load_dotenv()
logging.basicConfig(level=logging.INFO)
token = os.environ.get('HF_TOKEN')
huggingface_hub.login(token)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def get_model_param_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def get_label(data_item, dataset_name):
    if dataset_name == "ultrachat":
        return data_item["messages"][1]["content"]
    elif dataset_name == "truthfulqa":
        return data_item["best_answer"]

class AR_Trainer:
    def __init__(self, model_id, dataset, test_dataset, args):
        self.config = args
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set model (with LoRA training)
        self.model = LlamaForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False
        ).eval()

        # add special token <EMBED>
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.dataset = dataset
        self.test_dataset = test_dataset
        
        self.combined_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=1)
        
        self.test_dataloader = DataLoader(self.test_dataset)
       

    def eval(self):
        result_cache = []
        for item in tqdm(self.test_dataloader):
            prompt = get_prompt(item, self.config.dataset)
            label = get_label(item, self.config.dataset)
            
            generated_ids = self.generate_custom(prompt)
            generation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            logging.info(f"prompt: {prompt[0]}")
            logging.info(f"generation: {generation}")
            logging.info(f"gold response: {label[0]}")
            result_cache.append({
                "prompt": prompt[0], 
                "generation": generation, 
                "gold response": label[0]
            })
        
        with open(f"ar_result_{self.config.dataset}_train{self.config.train_size}_test{self.config.test_size}_gen{self.config.gen_length}.jsonl", 'w') as f:
            for item in result_cache:
                line = json.dumps(item)
                f.write(f"{line}\n")

    @torch.no_grad()
    def generate_custom(self, prompt, max_new_tokens=32, temperature=0.0, top_p=0.9):
        tokenizer = self.tokenizer
        model = self.model
        device = self.device
        
        # Prompt embeddings
        prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        
        output_ids = model.generate(prompt_ids, 
                                max_new_tokens=max_new_tokens, 
                                temperature=temperature, 
                                top_p=top_p,
                                do_sample=False,
                                return_dict_in_generate=False)
        gen_ids = output_ids[0][prompt_ids.shape[1]:]
        
        return gen_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    
    parser.add_argument("--dataset", type=str, choices=["truthfulqa", "wikitext", "ultrachat"])
    
    parser.add_argument("--train_size", type=int)
    parser.add_argument("--test_size", type=int)
    
    parser.add_argument("--gen_length", type=int)
    
    args = parser.parse_args()
    
    logging.info(args)
    
    torch.manual_seed(42)
    
    if args.dataset == "ultrachat":
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(args.train_size))
        test_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft").select(range(args.test_size))
    elif args.dataset == "wikitext":
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train").select(range(args.train_size))
        test_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test").select(range(args.test_size))
    elif args.dataset == "truthfulqa":
        # use all samples
        subset = "generation"
        split = "validation"
        dataset = load_dataset("truthfulqa/truthful_qa", subset, split=split)
        dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)
        dataset, test_dataset = dataset_dict["train"], dataset_dict["test"]
   
    trainer = AR_Trainer(args.model_id, dataset, test_dataset, args)
    trainer.eval()
   