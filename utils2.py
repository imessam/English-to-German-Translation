import torch
import math


def subsequent_mask(tokens, mode, pad_index, device="cpu"):
    """Mask out subsequent positions."""
    size = tokens.size(-1)
    attn_shape = (1, size, size)

    mask = (tokens != pad_index).unsqueeze(-2).to(device)
    # print(f"mask : {mask.shape}")
    if mode == "input":
        return mask

    sub_mask = (torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    ) == 0).to(device)
    # print(f"subsequent_mask : {subsequent_mask.shape}")
    # print(f"mask & subsequent_mask : {(mask & subsequent_mask).shape}")

    return mask & sub_mask


def infer(model, input_text, tokenizer, device="cpu"):
    model.eval()

    eos_idx = tokenizer.token_to_id("[EOS]")

    input_tokens = torch.tensor([tokenizer.encode(input_text).ids]).to(device)
    input_tokens = input_tokens[input_tokens != tokenizer.token_to_id("[BOS]")].view(
        (input_tokens.shape[0], input_tokens.shape[1] - 1))

    print(f"Input English Sentence : {tokenizer.decode(input_tokens[0].tolist())}")

    input_masks = subsequent_mask(input_tokens, pad_index=tokenizer.token_to_id("[PAD]"), mode="input", device=device)
    # print(f"input_masks : {input_masks.shape}")

    input_embeddings = model.embedding(input_tokens) * math.sqrt(model.d_model)
    # print(f"input embeddings : {input_embeddings.shape}")

    input_pos_embeddings = model.inp_pos_encoding(input_embeddings)
    # print(f"input_pos_embeddings : {input_pos_embeddings.shape}")

    input_encodings = model.encode(input_pos_embeddings, input_masks)
    # print(f"input_encodings : {input_encodings.shape}")

    output_tokens = torch.tensor([[tokenizer.token_to_id("[BOS]")]]).to(device).type_as(input_tokens)

    while (output_tokens[:, -1] != eos_idx) and (output_tokens.shape[-1] < 200):
        outputs_masks = subsequent_mask(output_tokens, pad_index=tokenizer.token_to_id("[PAD]"), mode="output",
                                        device=device)
        # print(f"outputs_masks : {outputs_masks.shape}")
        # print(f"outputs_masks : {outputs_masks}")

        ##Output embeddings##
        output_embeddings = model.embedding(output_tokens) * math.sqrt(model.d_model)
        # print(f"output embeddings : {output_embeddings.shape}")

        ##Add Positional Encoding##
        output_pos_embeddings = model.out_pos_encoding(output_embeddings)
        # print(f"output_pos_embeddings : {output_pos_embeddings.shape}")

        decodings = model.decode(output_pos_embeddings, input_encodings, input_masks, outputs_masks).to(device)
        # print(f"decodings : {decodings.shape}")
        # print(f"decodings : {decodings}")

        output_scores = model.finalLayer(decodings[:, -1])
        # print(f"output_scores : {output_scores.shape}")
        # print(f"output_scores : {output_scores}")

        output_proba = torch.softmax(output_scores, dim=-1)
        # print(f"output_proba : {output_proba.shape}")
        #         print(output_proba)

        next_word = torch.argmax(output_proba, dim=-1).unsqueeze(0)
        # print(f"next_word : {next_word.shape}")

        output_tokens = torch.cat([output_tokens, next_word], dim=-1)
        # print(f"output_tokens : {output_tokens.shape}")
        # print(output_tokens)

    return tokenizer.decode(output_tokens[0].tolist())
