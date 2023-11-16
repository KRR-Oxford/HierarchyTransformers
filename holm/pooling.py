import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    token_embeddings = model_output["last_hidden_state"]  # or [0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Max Pooling - Take the max value over time for every dimension.
def max_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    token_embeddings = model_output["last_hidden_state"]  # or [0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    token_embeddings[
        input_mask_expanded == 0
    ] = -1e9  # set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]


# [CLS] Pooling
def cls_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    token_embeddings = model_output["last_hidden_state"]  # or [0]
    # first dimension is the batch size
    return token_embeddings[:, 0]
