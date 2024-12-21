import torch

def get_distribution(logits, temperature):
    probs = torch.softmax(logits / (temperature + 1e-10), dim=-1)
    return probs

def sample(logits, temperature, top_p=0.7):
    """
    Sample from the logits distribution using temperature and nucleus sampling.
    
    Args:
        logits (torch.Tensor): Raw logits output from the model, shape [batch_size, vocab_size]
        temperature (float): Temperature for softmax scaling
        top_p (float): Cumulative probability threshold for nucleus sampling
        
    Returns:
        torch.Tensor: Sampled token indices, shape [batch_size]
    """
    if temperature == 0:
        # Greedy sampling
        return torch.argmax(logits, dim=-1)
        
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
        
    # Convert to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for top-p filtering
    mask = cumulative_probs <= top_p
    
    # Ensure at least one token is selected
    mask[..., 0] = True
    
    # Zero out probabilities not in top-p
    probs_masked = probs.clone()
    for batch_idx in range(logits.shape[0]):
        indices_to_remove = sorted_indices[batch_idx][~mask[batch_idx]]
        probs_masked[batch_idx, indices_to_remove] = 0
        
    # Renormalize probabilities
    probs_masked = probs_masked / probs_masked.sum(dim=-1, keepdim=True)
    
    # Sample from the filtered distribution
    return torch.multinomial(probs_masked, num_samples=1).squeeze(-1)

def sample_from_draft_model(model, head, initial_prompt_seq, init_target_hidden, step_size, temperature=1.0, top_p=0.7):
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    target_hidden = init_target_hidden.detach().clone()

    out_logits = []
    
    origin_device = target_hidden.device
    device = model.embed_tokens.weight.data.device
    target_hidden = target_hidden.to(device)
    fin_prompt_seq = fin_prompt_seq.to(device)

    for _ in range(step_size):
        # sample_token_logits = model(target_hidden, fin_prompt_seq).logits[:, -1, :]
        draft_hidden = model(target_hidden, fin_prompt_seq)
        sample_token_logits = head(draft_hidden)[:, -1, :]
        sample_token = sample(sample_token_logits, temperature=temperature, top_p=top_p)
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
        out_logits.append(sample_token_logits)
        target_hidden = torch.cat((target_hidden, draft_hidden[:, -1:, :]), dim=1)

    out_logits = torch.stack(out_logits, dim=1)
    return fin_prompt_seq.to(origin_device), out_logits.to(origin_device)
    