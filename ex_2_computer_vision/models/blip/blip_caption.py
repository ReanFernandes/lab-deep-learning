"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from torch.nn import functional as F

from models.bert.bert import BertLMHeadModel
from models.blip.blip import BlipBase
from models.blip.blip_config import BlipConfig
from models.vit.vit import VisionTransformerEncoder


class BlipCaption(BlipBase):
    def __init__(self, image_encoder, text_decoder, max_txt_len=40, cfg=None):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.visual_encoder = image_encoder
        self.text_decoder = text_decoder
        self.max_txt_len = max_txt_len
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg: BlipConfig):
        # vision encoder
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        # text encoder + multimodal decoder
        text_decoder = BertLMHeadModel.from_config(cfg.bert_cfg)
        model = cls(image_encoder, text_decoder, max_txt_len=cfg.max_txt_len)
        return model

    def forward(self, image, caption):
        #encode images
        image_feats = self.visual_encoder.forward_features(image)
        
        #encode text 
        text_feats = self.text_decoder(caption)

        #combine modalities 
        combined_feats = torch.cat((image_feats, text_feats), dim=1)

        # generate captions
        captions = self.generate(combined_feats)

        return captions



    def generate(
            self,
            image,
            use_topk_sampling=False,
            prompt="a picture of",
            min_length=10,
            max_length=30,
            topk=50,
            temperature=1.0,
    ):
        """

        Args:
            image: shape (batch_size, 3, H, W)
            use_topk_sampling: Whether to use nucleus sampling. If False, use top-k sampling.
            prompt: Prompt for decoding
            min_length: The minimum length of the sequence to be generated.
            max_length: The maximum length of the sequence to be generated.
            topk: Sample from top k tokens with highest probability when sampling.
            temperature: The value used to module the next token probabilities.

        Returns:
            captions (list): A list of strings of length batch_size
        """
        batch_size = image.shape[0]
        device = image.device

        # encode images
        # START TODO #################
        # Pass the image through the visual encoder to get the image embeddings
        image_embeds = self.visual_encoder.forward_features(image)
        # END TODO ###################
        image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long).to(device)
        validate_encoder_outputs(image_embeds, image_atts)

        # tokenize text, remove EOS token
        in_prompt = [prompt] * image_embeds.shape[0]
        tokenized_prompt = self.tokenizer(in_prompt, return_tensors="pt").to(device)
        input_ids, attention_mask = tokenized_prompt.input_ids, tokenized_prompt.attention_mask
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]
        validate_input_ids(input_ids, attention_mask, max_length)

        if not use_topk_sampling:
            # greedy search
            decoder_out = self.greedy_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                min_length=min_length,
                max_length=max_length,
            )
        else:
            # sampling
            decoder_out = self.sampling(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                min_length=min_length,
                max_length=max_length,
                topk=topk,
                temperature=temperature,
            )

        # decode text using tokenizer
        outputs = self.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)

        # remove the input prompt
        captions = [output[len(prompt):] for output in outputs]

        return captions

    def greedy_search(
            self,
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            min_length=10,
            max_length=30,
    ):
        eos_token_id = self.tokenizer.sep_token_id  # bert uses [SEP] token for end of sentence

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        while True:
            # START TODO #################
            # 1. forward pass through self.text_decoder to get the next token logits.
            # the text decoder is a model of type BertLMHeadModel defined in models/bert/bert.py
            # 2. select only the last token's output from the output logits, these are the logits
            # used for predicting the next token.
            decoder_out= self.text_decoder.forward(input_ids=input_ids, 
                                                   attention_mask=attention_mask, 
                                                   encoder_hidden_states=encoder_hidden_states, 
                                                   encoder_attention_mask=encoder_attention_mask)
            next_token_logits = decoder_out[:, -1, :]
            # END TODO ###################
            # logits shape (batch_size, current_sequence_len, vocab_size)
            # next_token_logits shape (batch_size, vocab_size)

            next_tokens_logits = enforce_min_length(
                input_ids, next_token_logits, min_length, eos_token_id)

            # START TODO #################
            # 1. select the token index with the maximum probability (argmax of the next
            # token scores) to get next_tokens shape (batch_size,)
            # 2. change the shape of next_tokens from (batch_size,) to (batch_size, 1) by adding
            # a new dimension
            # 3. use torch.cat to combine the current input_ids and the new tokens
            
            next_tokens = torch.argmax(next_tokens_logits, dim=-1)
            next_tokens = next_tokens.unsqueeze(1)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            # END TODO ###################

            # after extending the input_ids, also extend the attention_mask to match
            attention_mask = torch.cat([
                attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id).long()

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or input_ids.shape[-1] >= max_length:
                break

        return input_ids

    def sampling(
            self,
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            min_length=10,
            max_length=30,
            topk=50,
            temperature=1.0,
    ):
        eos_token_id = self.tokenizer.sep_token_id  # bert uses [SEP] token for end of sentence

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        while True:
            # START TODO #################
            # same as in function greedy_search above to get the next_token_logits
            decoder_out= self.text_decoder.forward(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
            next_token_logits = decoder_out[:, -1, :]
            # END TODO ###################

            # enforce minimum length
            next_tokens_logits = enforce_min_length(
                input_ids, next_token_logits, min_length, eos_token_id)

            # START TODO #################
            # 1. modify the next_token_logits as given by the temperature
            # 2. apply the topk sampling algorithm. useful commands are:
            # torch.topk, F.softmax, torch.multinomial, torch.gather
            # 3. same as above, add the new token to the input_ids
            
            
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            topk_probs = sorted_probs[:, :topk]
            topk_indices = sorted_indices[:, :topk]
            samples = torch.multinomial(topk_probs, 1)
            next_tokens = torch.gather(topk_indices, -1, samples)

            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # END TODO ###################

            # after extending the input_ids, also extend the attention_mask to match
            attention_mask = torch.cat([
                attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id).long()

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or input_ids.shape[-1] >= max_length:
                break

        return input_ids


def enforce_min_length(input_ids, next_token_logits, min_length, eos_token_id):
    """[`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0."""
    if min_length <= 1:
        return next_token_logits
    cur_len = input_ids.shape[-1]
    if cur_len < min_length:
        next_token_logits[:, eos_token_id] = -float("inf")
    return next_token_logits


def validate_input_ids(input_ids, attention_mask, max_length):
    assert input_ids.ndim == 2, "Input ids must be a 2D tensor (batch_size, seq_len)."
    assert attention_mask.shape == input_ids.shape, (
        "Attention mask must have the same shape as input ids.")
    input_ids_seq_length = input_ids.shape[-1]
    assert input_ids_seq_length < max_length, (
        f"Input sequence length {input_ids_seq_length} "
        f"is greater than max_length {max_length}")


def validate_encoder_outputs(encoder_hidden_states, encoder_attention_mask):
    assert encoder_hidden_states.ndim == 3, (
        "Encoder hidden states must be a 3D tensor (batch_size, encoder_seq_len, dim).")
    assert encoder_attention_mask.ndim == 2, (
        "Encoder attention mask must be a 2D tensor (batch_size, encoder_seq_len).")
    assert encoder_attention_mask.shape == encoder_hidden_states.shape[:2], (
        "Encoder attention mask must have the same shape as encoder hidden states.")
