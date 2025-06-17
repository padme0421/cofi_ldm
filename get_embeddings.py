import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, DataCollatorWithPadding
from generate import generate
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split

from collections import defaultdict
from datasets import load_dataset


def get_prompt(data_item, dataset_name):
    if dataset_name == "ultrachat":
        return data_item["prompt"]
    elif dataset_name == "truthfulqa":
        return data_item["question"]
    
def extract_llada_hidden(text, tokenizer, model, block_size=8):
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
        pad_len = (block_size - input_ids.size(0) % block_size) % block_size
        if pad_len > 0:
            input_ids = torch.cat([input_ids, input_ids.new_full((pad_len,), tokenizer.eos_token_id)])
        blocks = input_ids.view(-1, block_size)

        embs = []
        for blk in blocks:
            output = model(blk.unsqueeze(0), output_hidden_states=True)
            hs = output.hidden_states[-1].mean(1).squeeze(0)
            embs.append(hs)
        return torch.stack(embs)

@torch.no_grad()
def block_embeddings_hidden(model: PreTrainedModel, dataset: Dataset, test_dataset: Dataset,
                     tokenizer: PreTrainedTokenizer, config):
    
    embedding_list = []
    result_list = []
    for data in tqdm(dataset):
        prompt = get_prompt(data, config.dataset)

        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        embeddings = extract_llada_hidden(prompt, tokenizer, model, config.block_size)

        embedding_list.append({"block_embedding": embeddings})
            
    test_embedding_list = []
    test_result_list = []
    for data in tqdm(test_dataset):
        prompt = get_prompt(data, config.dataset)

        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        
        embeddings = extract_llada_hidden(prompt, tokenizer, model, config.block_size)

        test_embedding_list.append({"block_embedding": embeddings})
        
    embedding_dataset = Dataset.from_list(embedding_list)
    print(embedding_dataset[0])
    
    test_embedding_dataset = Dataset.from_list(test_embedding_list)
    print(test_embedding_dataset[0])
    
    signature = f"{config.dataset}_train{config.train_size}_test{config.test_size}_blk{config.block_size}_gen{config.gen_length}"
    
    path = f"/data/gahyunyoo/cofi_ldm/block_embeddings_llada_hidstate_{signature}"
    test_path = f"/data/gahyunyoo/cofi_ldm/test_block_embeddings_llada_hidstate_{signature}"
    
    embedding_dataset.save_to_disk(path)
    test_embedding_dataset.save_to_disk(test_path)
    
    with open(f"llada_output_{signature}_train.jsonl", 'w') as f:
        for line in result_list:
            f.write(f"{json.dumps({'generation': line})}\n")
    
    with open(f"llada_output_{signature}_test.jsonl", 'w') as f:
        for line in test_result_list:
            f.write(f"{json.dumps({'generation': line})}\n")
    
    return embedding_dataset, test_embedding_dataset

def block_embeddings(model: PreTrainedModel, dataset: Dataset, test_dataset: Dataset,
                     tokenizer: PreTrainedTokenizer, config):
    
    embedding_list = []
    result_list = []
    for data in tqdm(dataset):
        prompt = get_prompt(data, config.dataset)

        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        prompt_ids = tokenizer(prompt)['input_ids']
        prompt_ids = torch.tensor(prompt_ids).to(device).unsqueeze(0) # [batch_size, seq_len]
        print(f"prompt ids size: {prompt_ids.size()}")

        out = generate(model, prompt_ids, steps=config.gen_length, gen_length=config.gen_length, block_length=config.block_size, temperature=0., cfg_scale=0., remasking='low_confidence')
        response = out[:, prompt_ids.shape[1]:]
        
        response_text = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
        print(response_text)
        result_list.append(response_text)
        
        print(f"response size: {response.size()}")
    
        # for speed, embed at once
        embedding = model.get_input_embeddings()(response) # [batch_size, seq_len, dim]
        
        batch_blocks = torch.split(embedding, 1, dim=0) # batch_size blocks of size [seq_len, dim]
        for item_blocks in batch_blocks:
            item_blocks = item_blocks.squeeze(dim=0)
            blocks = torch.split(item_blocks, config.block_size, dim=0) # num_blocks blocks of size [block_size, dim]
            embedding_list.append({"block_embedding": [block.tolist() for block in blocks]})
            print(f"{len(blocks)} blocks of size {blocks[0].size()}")
            
    test_embedding_list = []
    test_result_list = []
    for data in tqdm(test_dataset):
        prompt = get_prompt(data, config.dataset)

        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        prompt_ids = tokenizer(prompt)['input_ids']
        prompt_ids = torch.tensor(prompt_ids).to(device).unsqueeze(0) # [batch_size, seq_len]
        print(f"prompt ids size: {prompt_ids.size()}")

        out = generate(model, prompt_ids, steps=config.gen_length, gen_length=config.gen_length, block_length=config.block_size, temperature=0., cfg_scale=0., remasking='low_confidence')
        response = out[:, prompt_ids.shape[1]:]
        
        response_text = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
        print(response_text)
        test_result_list.append(response_text)
        
        print(f"response size: {response.size()}")
    
        # for speed, embed at once
        embedding = model.get_input_embeddings()(response) # [batch_size, seq_len, dim]
        
        batch_blocks = torch.split(embedding, 1, dim=0) # batch_size blocks of size [seq_len, dim]
        for item_blocks in batch_blocks:
            item_blocks = item_blocks.squeeze(dim=0)
            blocks = torch.split(item_blocks, config.block_size, dim=0) # num_blocks blocks of size [block_size, dim]
            test_embedding_list.append({"block_embedding": [block.tolist() for block in blocks]})
            print(f"{len(blocks)} blocks of size {blocks[0].size()}")
        
    embedding_dataset = Dataset.from_list(embedding_list)
    print(embedding_dataset[0])
    
    test_embedding_dataset = Dataset.from_list(test_embedding_list)
    print(test_embedding_dataset[0])
    
    signature = f"{config.dataset}_train{config.train_size}_test{config.test_size}_blk{config.block_size}_gen{config.gen_length}"
    
    path = f"/data/gahyunyoo/cofi_ldm/block_embeddings_llada_{signature}"
    test_path = f"/data/gahyunyoo/cofi_ldm/test_block_embeddings_llada_{signature}"
    
    embedding_dataset.save_to_disk(path)
    test_embedding_dataset.save_to_disk(test_path)
    
    with open(f"llada_output_{signature}_train.jsonl", 'w') as f:
        for line in result_list:
            f.write(f"{json.dumps({'generation': line})}\n")
    
    with open(f"llada_output_{signature}_test.jsonl", 'w') as f:
        for line in test_result_list:
            f.write(f"{json.dumps({'generation': line})}\n")
    
    return embedding_dataset, test_embedding_dataset
            
if __name__ == "__main__":
    device = 'cuda'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    
    parser.add_argument("--dataset", type=str, choices=["wikitext", "ultrachat", "truthfulqa"])
    
    parser.add_argument("--train_size", type=int)
    parser.add_argument("--test_size", type=int)
    
    parser.add_argument("--block_size", type=int)
    
    parser.add_argument("--gen_length", type=int)
    
    args = parser.parse_args()

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
    
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    block_embeddings(model, dataset, test_dataset, tokenizer, args)
    
