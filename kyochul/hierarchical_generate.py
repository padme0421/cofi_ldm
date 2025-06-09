import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def hierarchical_generate(model, prompt, concept_steps=32, gen_length=128, 
                         concept_blocks=8, temperature=0., mask_id=126336):
    '''
    Hierarchical generation with two stages:
    1. Concept stage: Generate concept tokens using diffusion (broad thinking)
    2. Expansion stage: Expand each concept autoregressively (detailed thinking)
    
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        concept_steps: Diffusion steps for concept generation.
        gen_length: Total generated answer length.
        concept_blocks: Number of concept blocks to generate.
        temperature: Categorical distribution sampling temperature.
        mask_id: The token id of [MASK] is 126336.
    '''
    
    tokens_per_block = gen_length // concept_blocks
    device = model.device
    
    print(f"Stage 1: Generating {concept_blocks} concept blocks with {concept_steps} diffusion steps...")
    
    # === Stage 1: Concept Block Generation (Diffusion) ===
    # Initialize concept sequence with masks
    concept_x = torch.full((1, prompt.shape[1] + concept_blocks), mask_id, dtype=torch.long).to(device)
    concept_x[:, :prompt.shape[1]] = prompt.clone()
    
    # Diffusion process for concept generation
    for step in range(concept_steps):
        mask_index = (concept_x == mask_id)
        if not mask_index.any():
            break
            
        # Get model predictions
        logits = model(concept_x).logits
        logits_with_noise = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        
        # Calculate confidence scores
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(mask_index, x0_p, -np.inf)
        
        # Determine how many concepts to confirm this step
        remaining_masks = mask_index.sum().item()
        concepts_to_confirm = max(1, remaining_masks // (concept_steps - step))
        concepts_to_confirm = min(concepts_to_confirm, remaining_masks)
        
        if concepts_to_confirm > 0:
            # Select top confidence concepts to confirm
            _, top_indices = torch.topk(confidence.flatten(), concepts_to_confirm)
            # Convert flat indices to 2D indices
            batch_indices = top_indices // concept_x.shape[1]
            seq_indices = top_indices % concept_x.shape[1]
            
            # Update concept tokens
            concept_x[batch_indices, seq_indices] = x0[batch_indices, seq_indices]
        
        if step % 10 == 0:
            print(f"  Concept step {step}/{concept_steps}, remaining masks: {mask_index.sum().item()}")
    
    print(f"Stage 2: Expanding each concept block autoregressively...")
    
    # === Stage 2: Autoregressive Expansion ===
    final_tokens = prompt.clone()
    
    for block_idx in range(concept_blocks):
        print(f"  Expanding block {block_idx + 1}/{concept_blocks}...")
        
        # Get the concept token for this block
        concept_token_idx = prompt.shape[1] + block_idx
        if concept_token_idx < concept_x.shape[1]:
            concept_token = concept_x[0, concept_token_idx:concept_token_idx + 1]
            
            # Start with current sequence + concept token
            current_sequence = torch.cat([final_tokens, concept_token], dim=1)
            
            # Generate remaining tokens in this block autoregressively
            for token_pos in range(1, tokens_per_block):
                # Get next token prediction
                logits = model(current_sequence).logits
                next_token_logits = logits[0, -1, :]  # Last position logits
                
                # Apply temperature if specified
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    # Sample from distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    # Greedy selection
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Add next token to sequence
                current_sequence = torch.cat([current_sequence, next_token.unsqueeze(0)], dim=1)
            
            # Add the generated block to final result (excluding the concept token)
            block_tokens = current_sequence[0, final_tokens.shape[1] + 1:]  # Skip concept token
            final_tokens = torch.cat([final_tokens, block_tokens.unsqueeze(0)], dim=1)
        else:
            # If no concept token available, generate empty block
            empty_block = torch.full((1, tokens_per_block), mask_id, dtype=torch.long).to(device)
            final_tokens = torch.cat([final_tokens, empty_block], dim=1)
    
    return final_tokens


@torch.no_grad()
def original_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                     cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Original generation method from the provided code
    '''
    def get_num_transfer_tokens(mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1
        return num_transfer_tokens

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model
    m = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    print(f"\nInput prompt: {prompt}")
    print(f"Input token length: {input_ids.shape[1]}")

    # Method 1: Hierarchical generation (our new method)
    print("\n" + "="*50)
    print("HIERARCHICAL GENERATION (NEW METHOD)")
    print("="*50)
    
    hierarchical_out = hierarchical_generate(
        model, 
        input_ids, 
        concept_steps=32,      # Steps for concept generation
        gen_length=128,        # Total generation length
        concept_blocks=8,      # Number of concept blocks
        temperature=0.        # Deterministic generation
    )
    
    hierarchical_response = tokenizer.batch_decode(hierarchical_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"\nHierarchical Response:")
    print(hierarchical_response)

    # Method 2: Original generation (for comparison)
    print("\n" + "="*50)
    print("ORIGINAL GENERATION (BASELINE)")
    print("="*50)
    
    original_out = original_generate(
        model, 
        input_ids, 
        steps=128, 
        gen_length=128, 
        block_length=32, 
        temperature=0., 
        cfg_scale=0., 
        remasking='low_confidence'
    )
    
    original_response = tokenizer.batch_decode(original_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"\nOriginal Response:")
    print(original_response)

    print("\n" + "="*50)
    print("COMPARISON COMPLETE")
    print("="*50)


if __name__ == '__main__':
    main()