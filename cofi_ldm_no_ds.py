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

load_dotenv()
logging.basicConfig(level=logging.INFO)
token = os.environ.get('HF_TOKEN')
huggingface_hub.login(token)

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class CoFi_LDM_Trainer:
    def __init__(self, model_id: str, dataset: Dataset, embed_dataset: Dataset, 
                 test_dataset: Dataset, test_embed_dataset: Dataset):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LlamaForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_cache=False
        )
        self.model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        self.dataset = dataset
        self.embed_dataset = embed_dataset
        self.test_dataset = test_dataset
        self.test_embed_dataset = test_embed_dataset

        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    @torch.no_grad()
    def generate_custom(self, prompt: str, response_embeddings,
                        temperature=1.0, top_p=0.9) -> torch.Tensor:

        tokenizer = self.tokenizer
        model = self.model
        
        num_blocks, block_size, _ = response_embeddings.shape
        
        prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        prompt_embeds = model.get_input_embeddings()(prompt_ids).expand(num_blocks, -1, -1)
        
        response_embeddings = torch.tensor(response_embeddings, device=self.device, dtype=torch.bfloat16)

        response_tokens = tokenizer(f"response_embeddings: ", return_tensors='pt').input_ids.to(self.device)
        response_tokens_embeds = model.get_input_embeddings()(response_tokens) # [tokens_len, dim]
        response_tokens_embeds = response_tokens_embeds.expand(num_blocks, -1, -1) # [num_blocks, tokens_len, dim]

        generated = torch.full((num_blocks, 1), tokenizer.bos_token_id, device=self.device)
        past_key_values = None

        for _ in range(block_size):
            next_input_ids = generated[:, -1:]
            generated_embeds = model.get_input_embeddings()(next_input_ids)

            outputs = model(
                inputs_embeds=torch.cat([prompt_embeds, response_tokens_embeds, response_embeddings, generated_embeds], dim=1),
                past_key_values=past_key_values,
                use_cache=True
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            generated = torch.cat([generated, next_token], dim=-1)

            if (next_token == tokenizer.eos_token_id).all():
                break

        return torch.flatten(generated)

    def training_step(self, inputs):
        tokenizer = self.tokenizer
        model = self.model

        response_embeddings = torch.tensor(inputs["response_embeddings"], dtype=torch.bfloat16, device=self.device)
        prompt_ids = tokenizer(f"prompt: {inputs['prompt']}", return_tensors='pt', padding=True).input_ids.to(self.device)
        gold_response_ids = tokenizer(f"response: {inputs['response']}", return_tensors='pt').input_ids.squeeze(0).to(self.device)

        num_blocks, block_size, _ = response_embeddings.shape

        gold_response_ids_padded = torch.zeros(num_blocks * block_size, dtype=torch.long, device=self.device)
        gold_response_ids_padded[:min(gold_response_ids.shape[0], gold_response_ids_padded.shape[0])] = \
            gold_response_ids[:gold_response_ids_padded.shape[0]]
        gold_response_ids = gold_response_ids_padded

        embedding_module = model.get_input_embeddings()
        prompt_embeds = embedding_module(prompt_ids).expand(num_blocks, -1, -1)

        gold_blocks = gold_response_ids[:num_blocks * block_size].reshape(num_blocks, block_size)
        ar_input_ids = gold_blocks[:, :-1]
        ar_labels = gold_blocks[:, 1:]

        ar_embeds = embedding_module(ar_input_ids)
        
        response_tokens = tokenizer(f"response_embeddings: ", return_tensors='pt').input_ids.to(self.device)
        response_tokens_embeds = embedding_module(response_tokens) # [tokens_len, dim]
        response_tokens_embeds = response_tokens_embeds.expand(num_blocks, -1, -1) # [num_blocks, tokens_len, dim]
        
        full_input_embeds = torch.cat([prompt_embeds, response_tokens_embeds, response_embeddings, ar_embeds], dim=1) # [num_blocks, block_size, dim]

        context_len = prompt_embeds.size(1) + block_size
        target_len = ar_labels.size(1)

        full_labels = torch.full(
            (num_blocks, full_input_embeds.size(1)),
            -100,
            dtype=torch.long,
            device=self.device
        )
        full_labels[:, -target_len:] = ar_labels

        output = model(inputs_embeds=full_input_embeds, labels=full_labels)
        return output.loss

    def train(self):
        torch.manual_seed(42)
        self.model.train()

        combined_items = []
        for text_item, embed_item in zip(self.dataset, self.embed_dataset):
            combined_items.append({
                "prompt": text_item["prompt"],
                "response": text_item["messages"][1]["content"],
                "response_embeddings": embed_item["block_embedding"]
            })

        combined_dataset = Dataset.from_list(combined_items)
        dataloader = DataLoader(combined_dataset, batch_size=1)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        scaler = torch.cuda.amp.GradScaler()

        for i, item in tqdm(enumerate(dataloader), total=len(dataloader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = self.training_step(item)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logging.info(f"step={i}, loss={loss.item():.4f}")

    def eval(self):
        dataloader = DataLoader(self.test_dataset)
        embed_dataloader = DataLoader(self.test_embed_dataset)
        for item, embed_item in tqdm(zip(dataloader, embed_dataloader)):
            generated_ids = self.generate_custom(item["prompt"], embed_item["block_embedding"])
            generation = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            logging.info(f"prompt: {item['prompt']}")
            logging.info(f"generation: {generation}")
            logging.info(f"gold response: {item['messages'][1]['content']}")

if __name__ == "__main__":

    model_id = "meta-llama/Llama-3.1-8B"
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(500))
    test_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft").select(range(5))
    embed_dataset = load_from_disk("/data/gahyunyoo/cofi_ldm/block_embeddings")
    test_embed_dataset = load_from_disk("/data/gahyunyoo/cofi_ldm/test_block_embeddings")

    trainer = CoFi_LDM_Trainer(model_id, dataset, embed_dataset, test_dataset, test_embed_dataset)
    for epoch in range(5):
        trainer.train()
        trainer.eval()
