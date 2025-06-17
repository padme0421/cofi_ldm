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

class CoFi_LDM_Trainer:
    def __init__(self, model_id, dataset, embed_dataset, test_dataset, test_embed_dataset, args):
        self.config = args
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set model (with LoRA training)
        self.model = LlamaForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False
        )

        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        
        self.model.get_input_embeddings().weight.requires_grad = True
        self.model.lm_head.weight.requires_grad = True
        
        self.model.print_trainable_parameters()
        self.model.train()

        # add special token <EMBED>
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if '<EMBED>' not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<EMBED>']})
            self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

        if self.config.mlp:
            self.block_embed_dim = 4096
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.block_embed_dim, 1024, dtype=torch.bfloat16),
                torch.nn.GELU(),
                torch.nn.Linear(1024, self.block_embed_dim, dtype=torch.bfloat16),
                torch.nn.GELU()
            )
            self.mlp.to(self.device)
            self.mlp.train()

        self.dataset = dataset
        self.embed_dataset = embed_dataset
        self.test_dataset = test_dataset
        self.test_embed_dataset = test_embed_dataset
        
        # combined items
        combined_items = []
        for text_item, embed_item in zip(self.dataset, self.embed_dataset):
            combined_items.append({
                "prompt": get_prompt(text_item, self.config.dataset), 
                "response": get_label(text_item, self.config.dataset),
                "response_embeddings": embed_item["block_embedding"]
            })

        # combined dataset
        combined_dataset = Dataset.from_list(combined_items).shuffle()
        self.combined_dataloader = DataLoader(combined_dataset, shuffle=True, batch_size=1)
        
        self.test_dataloader = DataLoader(self.test_dataset)
        self.test_embed_dataloader = DataLoader(self.test_embed_dataset)
        
        # set optimizer
        if self.config.mlp:
            self.optimizer = torch.optim.AdamW(
                list(self.mlp.parameters()) + [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.lr
            )
        else:
            self.optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.lr
            )
        
    def training_step(self, inputs):
        logging.info("training step")
        
        # Step 1: Response embeddings
        response_embeddings = torch.as_tensor(inputs["response_embeddings"], device=self.device, dtype=torch.bfloat16)
        block_num, block_size, embed_dim = response_embeddings.size()
        logging.info(f"response_embeddings.size: {response_embeddings.size()}")
        
        if self.config.mlp:
            compressed_response_embeddings = self.mlp(response_embeddings)
            logging.info(f"compressed_response_embeddings.size: {compressed_response_embeddings.size()}")
        else: 
            compressed_response_embeddings = response_embeddings

        # Step 2: Prompt embeddings
        prompt_ids = self.tokenizer(inputs['prompt'], return_tensors='pt').input_ids.to(self.device)
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids).expand(block_num, -1, -1)  # [block_num, prompt_len, embed_dim]
        logging.info(f"prompt_embeds.size: {prompt_embeds.size()}")

        # Step 3: Gold response embeddings and IDs
        gold_response_ids = self.tokenizer(inputs['response'], return_tensors='pt').input_ids.squeeze(0).to(self.device)  # [block_num * block_size]
           
        # Fill truncated positions with PAD instead of -100 before embedding
        pad_token_id = self.tokenizer.pad_token_id or 0  # fallback to 0 if pad_token_id is None
        trunc_gold_response_ids = torch.full((block_num * block_size,), pad_token_id, device=self.device)
        gold_length = min(gold_response_ids.size(0), block_num * block_size)
        
        trunc_gold_response_ids[:gold_length] = gold_response_ids[:gold_length] #[block_num * block_size]
        
        gold_response_embeds = self.model.get_input_embeddings()(trunc_gold_response_ids)  # [block_num * block_size, embed_dim]
        split_gold_response_embeds = gold_response_embeds.view(block_num, block_size, embed_dim)
        logging.info(f"split_gold_response_embeds.size: {split_gold_response_embeds.size()}")

        # Step 4: <EMBED> token
        embed_special_token_id = self.tokenizer("<EMBED>", return_tensors='pt').input_ids.to(self.device) # [1,1]
        embed_special_token_embeds = self.model.get_input_embeddings()(embed_special_token_id).expand(block_num, -1, -1)  # (embed) [1, 1, embed_dim] -> (expand) [block_num, 1, embed_dim]

        # Step 5: input_embeds = [<EMBED>, response, <EMBED>, prompt, gold_response]
        input_embeds = torch.cat([
            embed_special_token_embeds,      # [block_num, 1, embed_dim]
            compressed_response_embeddings,             # [block_num, block_size, embed_dim]
            embed_special_token_embeds,      # [block_num, 1, embed_dim]
            prompt_embeds,                   # [block_num, prompt_len, embed_dim]
            split_gold_response_embeds       # [block_num, block_size, embed_dim]
        ], dim=1)
        logging.info(f"input_embeds.size: {input_embeds.size()}")

        # Step 6: Labels
        split_gold_response_ids = trunc_gold_response_ids.view(block_num, block_size)  # [block_num, block_size]
        total_len = input_embeds.size(1)
        labels = torch.full((block_num, total_len), -100, dtype=torch.long, device=self.device)
        labels[:, -block_size:] = split_gold_response_ids # supervise each block 

        # Step 7: Forward pass
        output = self.model(inputs_embeds=input_embeds, labels=labels)

        return output.loss


    def train(self):
        
        for i, batch in tqdm(enumerate(self.combined_dataloader), total=len(self.combined_dataloader)):
            self.optimizer.zero_grad()
            
            try:
                loss = self.training_step(batch)
            except Exception as e:
                # Code to handle other exceptions
                print(f"An error occurred: {e}")
                continue
            
            before = get_model_param_norm(self.model)
            
            loss.backward()
            self.optimizer.step()
            
            after = get_model_param_norm(self.model)

            logging.info(f"step={i}, loss={loss:.4f}, norm_delta={after - before:.6f}")

    def eval(self):
        result_cache = []
        for item, embed_item in tqdm(zip(self.test_dataloader, self.test_embed_dataloader)):
            prompt = get_prompt(item, self.config.dataset)
            label = get_label(item, self.config.dataset)
            
            generated_ids = self.generate_custom(prompt, embed_item["block_embedding"])
            generation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            logging.info(f"prompt: {prompt[0]}")
            logging.info(f"generation: {generation}")
            logging.info(f"gold response: {label[0]}")
            result_cache.append({
                "prompt": prompt[0], 
                "generation": generation, 
                "gold response": label[0]
            })
        
        with open(f"fixed_result_{self.config.dataset}_train{self.config.train_size}_test{self.config.test_size}_blk{self.config.block_size}_gen{self.config.gen_length}_epochs{self.config.epochs}_lr{self.config.lr}_mlp={self.config.mlp}.jsonl", 'w') as f:
            for item in result_cache:
                line = json.dumps(item)
                f.write(f"{line}\n")

    @torch.no_grad()
    def generate_custom(self, prompt, response_embeddings, max_new_tokens=32, temperature=0.0, top_p=0.9):
        tokenizer = self.tokenizer
        model = self.model
        device = self.device

        # Convert response embeddings
        response_embeddings = torch.as_tensor(response_embeddings, device=device, dtype=torch.bfloat16)
        block_num, block_size, embed_dim = response_embeddings.size()
        
        if self.config.mlp:
            compressed_response_embeddings = self.mlp(response_embeddings)
        else:
            compressed_response_embeddings = response_embeddings

        # <EMBED> token
        embed_token_id = tokenizer("<EMBED>", return_tensors='pt').input_ids.to(device)
        embed_token_embed = model.get_input_embeddings()(embed_token_id).expand(block_num, -1, -1)  # [block_num, 1, embed_dim]

        # Prompt embeddings
        prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        prompt_embeds = model.get_input_embeddings()(prompt_ids).expand(block_num, -1, -1)

        # Precomputed static context: [<EMBED>, response, <EMBED>, prompt]
        static_context = torch.cat([
            embed_token_embed,
            compressed_response_embeddings,
            embed_token_embed,
            prompt_embeds
        ], dim=1)  # shape: [block_num, ctx_len, embed_dim]

        # Prepare initial generated IDs
        generated_ids = torch.full((block_num, 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
        
        for i in range(max_new_tokens):
            # Embed generated tokens
            gen_embeds = model.get_input_embeddings()(generated_ids)  # [block_num, gen_len, embed_dim]

            # Build full input: static context + generated so far
            input_embeds = torch.cat([static_context, gen_embeds], dim=1)  # [block_num, total_len, embed_dim]
            
            # Forward pass
            outputs = model(inputs_embeds=input_embeds)
            next_token_logits = outputs.logits[:, -1, :]  # last position

            # Sample next token
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)  # shape: [block_num]
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
        
            # Append to generated
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=1)
            
            # Stop if all have generated <eos>
            if (next_token == tokenizer.eos_token_id).all():
                break

        return torch.flatten(generated_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    
    parser.add_argument("--dataset", type=str, choices=["truthfulqa", "wikitext", "ultrachat"])
    
    parser.add_argument("--train_size", type=int)
    parser.add_argument("--test_size", type=int)
    
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gen_length", type=int)
    parser.add_argument("--mlp", action="store_true")
    
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
    
    default_train_size = 500 
    default_test_size = 5
    
    signature = f"{args.dataset}_train{args.train_size}_test{args.test_size}_blk{args.block_size}_gen{args.gen_length}"
    
    if (args.train_size == default_train_size) and (args.test_size == default_test_size):
        embed_dataset = load_from_disk(f"/data/gahyunyoo/cofi_ldm/block_embeddings_llada")
        test_embed_dataset = load_from_disk(f"/data/gahyunyoo/cofi_ldm/test_block_embeddings_llada")
    else:
        embed_dataset = load_from_disk(f"/data/gahyunyoo/cofi_ldm/block_embeddings_llada_{signature}")
        test_embed_dataset = load_from_disk(f"/data/gahyunyoo/cofi_ldm/test_block_embeddings_llada_{signature}")

    trainer = CoFi_LDM_Trainer(args.model_id, dataset, embed_dataset, test_dataset, test_embed_dataset, args)
    trainer.eval()
    for epoch in range(args.epochs):
        trainer.train()
        trainer.eval()
