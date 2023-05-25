# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Based on huggingface transformers/generation/utils.py"""
import torch


class Generator:
    def __init__(
        self,
        model,
        use_nucleus_sampling=False,
        num_beams=1,
        min_length=10,
        max_length=30,
        top_p=0.9,
        repetition_penalty=1.0,
        num_captions=1,
        temperature=1.0,
        pad_token_id=0,
        eos_token_id=102,
    ):
        self.model = model
        self.use_nucleus_sampling = use_nucleus_sampling
        self.num_beams = num_beams
        self.min_length = min_length
        self.max_length = max_length
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_captions = num_captions
        self.temperature = temperature
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        assert self.num_beams == 1, "Beam search not supported yet."
        assert self.repetition_penalty == 1.0, "Repetition penalty not supported yet."
        assert self.num_captions == 1, "Multiple captions not supported yet."

    def generate(
        self,
        input_ids,  # tokenized input prompt (batch_size, seq_len) "an image of "
        attention_mask,  # attention mask for input_ids (batch_size, seq_len)
        encoder_hidden_states,
        # image features (batch_size, n_patches_h * n_patches_w + 1, dim)
        encoder_attention_mask,
        # image attention mask (batch_size, n_patches_h * n_patches_w + 1)
    ):
        outputs = self.greedy_search(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return outputs

    def greedy_search(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,

    ):
        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        while True:
            # forward pass to get next token
            logits, hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            next_token_logits = logits[:, -1, :]

            # preprocess the logit distribution
            next_tokens_scores = self.enforce_min_length(input_ids, next_token_logits)

            # greedily select next token
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences * (next_tokens != self.eos_token_id).long()

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or input_ids.shape[-1] >= self.max_length:
                break

        return input_ids

    def enforce_min_length(self, input_ids, next_token_logits):
        """[`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0."""
        if self.min_length <= 1:
            return next_token_logits
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            next_token_logits[:, self.eos_token_id] = -float("inf")
        return next_token_logits
