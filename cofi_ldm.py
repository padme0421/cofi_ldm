from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, Trainer
from datasets import load_dataset, load_from_disk
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import torch
from typing import List
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def sample_top_p(probs, p):
    # source: llama3/generation
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class CoFi_LDM_Trainer:
    def __init__(self, model_id: str, dataset: Dataset, embed_dataset: Dataset):
        self.model = LlamaForCausalLM.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.dataset = dataset
        self.embed_dataset = embed_dataset
        
    @torch.no_grad()
    def generate_custom(self, prompt: str, response_embeddings: List[torch.Tensor],
                        temperature=1.0, top_p=0.9) -> torch.Tensor:
        '''
        parameters
            prompt: full prompt text
            response_embeddings: block embeddings for the whole response
            
        return:
            generation: tensor
        '''
        
        tokenizer = self.tokenizer
        model = self.model
        
        prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device) # [1, prompt_len]
        past_key_values = None
        block_size = response_embeddings[0].shape[1]
        
        # Prepare prompt embeddings (shared across blocks)
        prompt_embeds = model.get_input_embeddings()(prompt_ids)  # [1, prompt_len, dim]
        prompt_embeds = prompt_embeds.expand(num_blocks, -1, -1)  # [num_blocks, prompt_len, dim]
        
        # Prepare response blocks
        response_embeddings = torch.stack(response_embeddings, dim=0).to(model.device)  # [num_blocks, block_size, dim]
        num_blocks, block_size, embed_dim = response_embeddings.shape
        
        generated = torch.full((num_blocks, 1), tokenizer.bos_token_id)
      
        for i in range(block_size):
            # Only feed the last token (efficient with KV cache)
            
            next_input_ids = generated[:,-1:] # [num_blocks, generated_len]
            generated_embeds = self.model.get_input_embeddings()(next_input_ids)  # [num_blocks, generated_len, dim]
           
            outputs = self.model(
                inputs_embeds=torch.cat([prompt_embeds, response_embeddings, generated_embeds], dim=1),
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs.logits
            past_key_values = outputs.past_key_values  # update cache

            # Get the logits of the last token
            next_token_logits = logits[:, -1, :]

            # Apply temperature and top-p sampling
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Append to sequence
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            # Optionally: break on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        generated = torch.flatten(generated)
        return generated
    
    def training_step(self, inputs):
        """
        Fully parallel block-wise autoregressive training.
        inputs:
            - "prompt": str
            - "response": str
            - "response_embeddings": list of shape [num_blocks, block_size, embed_dim]
        """
        print("training step")
        response_embeddings = torch.tensor(inputs["response_embeddings"], dtype=torch.float)
        print(response_embeddings.size())
        
        tokenizer = self.tokenizer
        model = self.model

        # Tokenize prompt and response
        prompt_ids = tokenizer(inputs["prompt"], return_tensors='pt').input_ids.to(model.device)  # [1, prompt_len]
        gold_response_ids = tokenizer(inputs["response"], return_tensors='pt').input_ids.to(model.device).squeeze(0)  # [response_len]
       
        # Prepare response blocks
        num_blocks, block_size, embed_dim = response_embeddings.shape
        
        gold_response_ids_padded = torch.zeros(num_blocks * block_size, dtype=torch.int32)
        if gold_response_ids.shape[0] > (num_blocks * block_size):
            gold_response_ids_padded = gold_response_ids[:num_blocks * block_size]
        else:
            gold_response_ids_padded[:gold_response_ids.shape[0]] = gold_response_ids
        gold_response_ids = gold_response_ids_padded
        # Prepare prompt embeddings (shared across blocks)
        prompt_embeds = model.get_input_embeddings()(prompt_ids)  # [1, prompt_len, dim]
        prompt_embeds = prompt_embeds.expand(num_blocks, -1, -1)  # [num_blocks, prompt_len, dim]
        assert prompt_embeds.dtype == torch.float
        
        # Slice response tokens into blocks
        gold_blocks = gold_response_ids[:num_blocks * block_size].reshape(num_blocks, block_size)  # [num_blocks, block_size]

        # AR input: tokens 0 to K-2
        ar_input_ids = gold_blocks[:, :-1]  # [num_blocks, block_size - 1]
        ar_labels = gold_blocks[:, 1:]      # [num_blocks, block_size - 1]

        # Embeddings for AR input tokens
        ar_embeds = model.get_input_embeddings()(ar_input_ids)  # [num_blocks, block_size - 1, dim]
        assert ar_embeds.dtype == torch.float

        # Concatenate context: [prompt | response_embedding | gold_block_input]
        full_input_embeds = torch.cat([prompt_embeds, response_embeddings, ar_embeds], dim=1)  # [num_blocks, *, dim]

        # Compute token positions to predict
        context_len = prompt_embeds.size(1) + block_size  # prompt + response_embedding
        target_len = ar_labels.size(1)  # block_size - 1

        # Create full label tensor
        full_labels = torch.full(
            (num_blocks, full_input_embeds.size(1)),
            -100,
            dtype=torch.long,
            device=model.device
        )
        full_labels[:, -target_len:] = ar_labels  # only predict last block

        # Forward pass
        output = model(inputs_embeds=full_input_embeds, labels=full_labels)

        return output.loss

    
    def train(self):
        '''full training loop'''
        
        # configure optimizer
        optimizer = Adam(
                         self.model.parameters(),
                         eps=1e-06,
                         betas=(0.9, 0.98),
                         weight_decay=1e-05,
                         )
        
        self.model.train()
        
        combined_items = []
        
        for text_item, embed_item in zip(self.dataset, self.embed_dataset):
            combined_item = {
                "prompt": text_item["prompt"], 
                "response": text_item["messages"][1]["content"], 
                "response_embeddings": embed_item["block_embedding"]
            }
            combined_items.append(combined_item)
            
        combined_dataset = Dataset.from_list(combined_items)
        
        dataloader = DataLoader(combined_dataset)
        for i, item in tqdm(enumerate(dataloader)): # TODO: support batching
            loss = self.training_step(item)
            logging.info(loss)
            loss.backward()
            optimizer.step()
            
    def eval():
        pass
        
if __name__ == "__main__":
    
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # load original dataset
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    dataset = dataset.select(range(50))

    # load embeddings dataset 
    embed_dataset = load_from_disk("/data/gahyunyoo/cofi_ldm/block_embeddings")
    
    trainer = CoFi_LDM_Trainer(model_id, dataset, embed_dataset)
    
    trainer.train()
    
    trainer.eval()