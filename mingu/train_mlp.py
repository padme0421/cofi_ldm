import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Define MLP Mapper
class EmbeddingMapper(nn.Module):
    def __init__(self, input_dim=4096, output_dim=2560):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# Dataset class for UltraChat
class MLPTrainDataset(Dataset):
    def __init__(self, hf_ds, phi_tokenizer, max_length=128):
        self.ds = hf_ds
        self.phi_tok = phi_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        entry = self.ds[idx]
        messages = entry.get("messages", [])
        text = "\n".join([msg["content"] for msg in messages if msg["role"] in ("user", "assistant")])
        encoding = self.phi_tok(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        return {"text": text, "input_ids": input_ids}

# Extract LLaDA embeddings
@torch.no_grad()
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizers and models
    llada_tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    llada_model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True, output_hidden_states=True).eval().to(device)

    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").eval().to(device)

    mlp = EmbeddingMapper(input_dim=4096, output_dim=phi_model.config.hidden_size).to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-4)

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    train_ds = MLPTrainDataset(dataset, phi_tokenizer)
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    num_epochs = 3
    save_path = "checkpoints/mlp_epoch_3.pt"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            text = batch["text"][0]
            target_ids = batch["input_ids"].to(device)

            llada_emb = extract_llada_hidden(text, llada_tokenizer, llada_model).to(device)
            mapped = mlp(llada_emb).unsqueeze(0)  # [1, seq_len, hidden]
            labels = target_ids.unsqueeze(0)[:, 1:1 + mapped.shape[1]]

            outputs = phi_model(inputs_embeds=mapped, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")

        torch.save(mlp.state_dict(), save_path)
        print(f"✅ Saved MLP at {save_path}")

if __name__ == "__main__":
    main()

