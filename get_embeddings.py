import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, DataCollatorWithPadding
from generate import generate
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json

def block_embeddings(model: PreTrainedModel, dataset: Dataset, test_dataset: Dataset,
                     tokenizer: PreTrainedTokenizer, config):
    
    embedding_list = []
    result_list = []
    for data in tqdm(dataset):
        prompt = data['prompt']

        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        prompt_ids = tokenizer(prompt)['input_ids']
        prompt_ids = torch.tensor(prompt_ids).to(device).unsqueeze(0) # [batch_size, seq_len]
        print(f"prompt ids size: {prompt_ids.size()}")

        out = generate(model, prompt_ids, steps=32, gen_length=32, block_length=config.block_size, temperature=0., cfg_scale=0., remasking='low_confidence')
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
        prompt = data['prompt']

        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        prompt_ids = tokenizer(prompt)['input_ids']
        prompt_ids = torch.tensor(prompt_ids).to(device).unsqueeze(0) # [batch_size, seq_len]
        print(f"prompt ids size: {prompt_ids.size()}")

        out = generate(model, prompt_ids, steps=32, gen_length=32, block_length=config.block_size, temperature=0., cfg_scale=0., remasking='low_confidence')
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
    
    path = f"/data/gahyunyoo/cofi_ldm/block_embeddings_llada_{args.dataset}_{args.train_size}"
    test_path = f"/data/gahyunyoo/cofi_ldm/test_block_embeddings_llada_{args.dataset}_{args.test_size}"
    
    embedding_dataset.save_to_disk(path)
    test_embedding_dataset.save_to_disk(test_path)
    
    with open(f"llada_output_train{config.train_size}_test{config.test_size}_blk{config.block_size}_train.txt", 'w') as f:
        f.writelines(result_list)
    
    with open(f"llada_output_train{config.train_size}_test{config.test_size}_blk{config.block_size}_test.txt", 'w') as f:
        f.writelines(test_result_list)
    
    return embedding_dataset, test_embedding_dataset
            
if __name__ == "__main__":
    device = 'cuda'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    
    parser.add_argument("--dataset", type=str, choices=["wikitext", "ultrachat"])
    
    parser.add_argument("--train_size", type=int)
    parser.add_argument("--test_size", type=int)
    
    parser.add_argument("--block_size", type=int)
    
    args = parser.parse_args()

    if args.dataset == "ultrachat":
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(args.train_size))
        test_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft").select(range(args.test_size))
    elif args.dataset == "wikitext":
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train").select(range(args.train_size))
        test_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="test").select(range(args.test_size))
    
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    block_embeddings(model, dataset, test_dataset, tokenizer, args)
    
