from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM

# Load LLaDA in eval mode
llada_tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
llada_model = AutoModel.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base",
    trust_remote_code=True,
    output_hidden_states=True,
    torch_dtype=torch.float16
).eval()

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16
).eval()

@torch.no_grad()
def extract_llada_block_embeddings(text, block_size=4):
    # 토큰화 및 블록 분할
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
    n_tokens = tokens.shape[0]

    # 패딩 추가
    pad_len = (block_size - n_tokens % block_size) % block_size
    if pad_len > 0:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        padding = torch.full((pad_len,), pad_id, dtype=torch.long)
        tokens = torch.cat([tokens, padding], dim=0)

    blocks = tokens.view(-1, block_size)  # [N_blocks, block_size]
    embeddings = []

    for block in blocks:
        input_ids = block.unsqueeze(0)  # [1, block_size]
        output = llada_model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = output.hidden_states  # tuple of [layer_0, ..., layer_N]

        # 예: 마지막 layer의 평균 임베딩 사용
        last_hidden = hidden_states[-1]  # [1, block_size, hidden_dim]
        mean_emb = last_hidden.mean(dim=1).squeeze(0)  # [hidden_dim]
        embeddings.append(mean_emb)

    return torch.stack(embeddings)  # [N_blocks, hidden_dim]


class EmbeddingMapper(nn.Module):
    def __init__(self, input_dim=4096, output_dim=2560):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_dim),
            nn.LayerNorm(output_dim),     # ✅ 안정화
            nn.Tanh()                     # ✅ 범위 제한 (-1~1)
        )

    def forward(self, x):
        return self.mlp(x)


@torch.no_grad()
def generate_from_llada(phi_model, phi_tokenizer, llada_embeddings, mlp_model, max_new_tokens=30):
    mapped = mlp_model(llada_embeddings)
    mapped = torch.clamp(mapped, min=-5, max=5)
    mapped = mapped.unsqueeze(0)  # [1, N, D]

    mapped = mlp_model(llada_embeddings).unsqueeze(0)

    if torch.isnan(mapped).any():
        raise ValueError("❌ NaN in mapped")
    if torch.isinf(mapped).any():
        raise ValueError("❌ Inf in mapped")

    print("✅ mapped OK", mapped.shape, mapped.min().item(), mapped.max().item())

    # 1. pad_token_id 설정 보정
    if phi_tokenizer.pad_token_id is None:
        phi_tokenizer.pad_token = phi_tokenizer.eos_token

    # 2. attention_mask 생성
    attention_mask = torch.ones((1, mapped.shape[1]), dtype=torch.long)

    # 3. generate 호출
    outputs = phi_model.generate(
        inputs_embeds=mapped,
        attention_mask=attention_mask,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=phi_tokenizer.pad_token_id,
        eos_token_id=phi_tokenizer.eos_token_id
    )

    return phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
text = "Hello! What is your name? My name is mingu kang in seoul national university. Nice to meet you."
# 1. 임베딩 추출
llada_embeddings = extract_llada_block_embeddings(text).half()

# 2. MLP 정의 및 dtype 일치
mlp_model = EmbeddingMapper(input_dim=llada_embeddings.shape[1]).half()

# 3. 텍스트 생성
generated_text = generate_from_llada(model, tokenizer, llada_embeddings, mlp_model)
print(generated_text)

