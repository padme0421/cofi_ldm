import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, DataCollatorWithPadding
from generate import generate
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def block_embeddings(model: PreTrainedModel, dataset: Dataset, tokenizer: PreTrainedTokenizer, 
                     block_size: int):
    
    embedding_list = []
    
    for data in tqdm(dataset):
        prompt = data["prompt"]
        gold_response = data["messages"][1]["content"]

        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        prompt_ids = tokenizer(prompt)['input_ids']
        prompt_ids = torch.tensor(prompt_ids).to(device).unsqueeze(0) # [batch_size, seq_len]
        print(f"prompt ids size: {prompt_ids.size()}")

        out = generate(model, prompt_ids, steps=8, gen_length=64, block_length=block_size, temperature=0., cfg_scale=0., remasking='low_confidence')
        response = out[:, prompt_ids.shape[1]:]
        
        print(tokenizer.batch_decode(response, skip_special_tokens=True))
        print(f"response size: {response.size()}")
    
        # for speed, embed at once
        embedding = model.get_input_embeddings()(response) # [batch_size, seq_len, dim]
        
        batch_blocks = torch.split(embedding, 1, dim=0) # batch_size blocks of size [seq_len, dim]
        for item_blocks in batch_blocks:
            item_blocks = item_blocks.squeeze(dim=0)
            blocks = torch.split(item_blocks, block_size, dim=0) # num_blocks blocks of size [block_size, dim]
            embedding_list.append({"block_embedding": [block.tolist() for block in blocks]})
            print(f"{len(blocks)} blocks of size {blocks[0].size()}")
        
    embedding_dataset = Dataset.from_list(embedding_list)
    print(embedding_dataset[0])
    embedding_dataset.save_to_disk("/data/gahyunyoo/cofi_ldm/block_embeddings")
    return embedding_dataset
            
if __name__ == "__main__":
    device = 'cuda'

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    dataset = dataset.select(range(50))

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    block_embeddings(model, dataset, tokenizer, 16)
    
