#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
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

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    "quiet": QUIET,
    "regular": REGULAR,
    "verbose": VERBOSE,
    "very_verbose": VERY_VERBOSE,
}

BAG_OF_WORDS_ARCHIVE_MAP = {
    "legal": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    "military": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    "monsters": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    "politics": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    "positive_words": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    "religion": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    "science": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    "space": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    "technology": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}


class ClassificationHead(torch.nn.Module):
    """Classification Head for Transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        logits = self.mlp(hidden_state)
        return logits


def to_var(x, requires_grad=False, volatile=False, device="cuda"):
    # TODO this function converts the Tensor x to a torch variable,
    # to enable autograd on the Tensor. Variable is deprecated, instead use
    # a regular torch Tensor with requires_grad=True
    if torch.cuda.is_available() and device == "cuda":
        x = x.cuda()
    elif device != "cuda":
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        one_hot_bows_vectors=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR,
        kl_penalty=0.2,
):
    # Originally past is a list of tuples of length 2, instead stack elements
    # in the tuple to produce a list of tensors.
    past = [torch.stack(attn_block) for attn_block in past]

    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
            tuple(past[0].shape[:-2])
            + tuple([window_length])
            + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
            tuple(past[0].shape[:-2])
            + tuple([curr_length - window_length])
            + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # Accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape

        all_outputs = model(last, past_key_values=perturbed_past)
        all_logits = all_outputs.logits
        hidden = all_outputs.hidden_states[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        # Compute the Objective

        # Get the unnormalized rewards for each action
        for one_hot_bow in one_hot_bows_vectors:
            bow_sum = torch.mm(probs, torch.t(one_hot_bow)).sum()
            rewards = torch.t(one_hot_bow).sum(dim=1) / bow_sum

        # Compute the KL ratios
        unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
        unpert_probs = (
            unpert_probs + SMALL_CONST *
            (unpert_probs <= SMALL_CONST).float().to(device).detach()
        )
        correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
            device).detach()
        corrected_probs = probs + correction.detach()

        ratios = corrected_probs / unpert_probs
        log_ratios = ratios.log()
        kl = (corrected_probs * log_ratios).sum()

        # Compute normalized rewards (according to Ziegler et al., 2019)
        norm_rewards = rewards - kl_scale * log_ratios

        # Calculate adaptive KL penalty for next round
        # K_beta = 0.1
        # target = 6  # Default value in ppo library
        # proportional_error = torch.clamp(kl / target - 1, -0.2, 0.2)
        # kl_penalty = kl_penalty * (1 + K_beta * proportional_error)
        # kl_penalty = kl_penalty.detach().item()

        # PPO clip objective
        epsilon = 0.2  # Best value according to PPO paper
        objective = (
            probs *
            torch.minimum(
                ratios * norm_rewards,
                torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * norm_rewards,
            )
        ).sum()

        if verbosity_level >= VERBOSE:
            print(" pplm_objective", objective.data.cpu().numpy())
        loss_per_iter.append(objective.data.cpu().numpy())

        # compute gradients
        objective.backward()

        # normalize gradients
        grad = [
            stepsize * p_.grad.data.cpu().numpy()
            for p_ in curr_perturbation
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    pert_past = [(attn_block[0], attn_block[1]) for attn_block in pert_past]

    return pert_past, grad_norms, loss_per_iter, kl_penalty

# def perturb_past(
#         past,
#         model,
#         last,
#         unpert_past=None,
#         unpert_logits=None,
#         accumulated_hidden=None,
#         grad_norms=None,
#         stepsize=0.01,
#         one_hot_bows_vectors=None,
#         classifier=None,
#         class_label=None,
#         loss_type=0,
#         num_iterations=3,
#         horizon_length=1,
#         window_length=0,
#         decay=False,
#         gamma=1.5,
#         kl_scale=0.01,
#         device='cuda',
#         verbosity_level=REGULAR
# ):
#     # Originally past is a list of tuples of length 2, instead stack elements
#     # in the tuple to produce a list of tensors.
#     past = [torch.stack(attn_block) for attn_block in past]

#     # Generate inital perturbed past
#     grad_accumulator = [
#         (np.zeros(p.shape).astype("float32"))
#         for p in past
#     ]

#     if accumulated_hidden is None:
#         accumulated_hidden = 0

#     if decay:
#         decay_mask = torch.arange(
#             0.,
#             1.0 + SMALL_CONST,
#             1.0 / (window_length)
#         )[1:]
#     else:
#         decay_mask = 1.0

#     # TODO fix this comment (SUMANTH)
#     # Generate a mask is gradient perturbated is based on a past window
#     _, _, _, curr_length, _ = past[0].shape

#     if curr_length > window_length and window_length > 0:
#         ones_key_val_shape = (
#             tuple(past[0].shape[:-2])
#             + tuple([window_length])
#             + tuple(past[0].shape[-1:])
#         )

#         zeros_key_val_shape = (
#             tuple(past[0].shape[:-2])
#             + tuple([curr_length - window_length])
#             + tuple(past[0].shape[-1:])
#         )

#         ones_mask = torch.ones(ones_key_val_shape)
#         ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
#         ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

#         window_mask = torch.cat(
#             (ones_mask, torch.zeros(zeros_key_val_shape)),
#             dim=-2
#         ).to(device)
#     else:
#         window_mask = torch.ones_like(past[0]).to(device)

#     # Accumulate perturbations for num_iterations
#     loss_per_iter = []
#     new_accumulated_hidden = None
#     for i in range(num_iterations):
#         if verbosity_level >= VERBOSE:
#             print("Iteration ", i + 1)
#         curr_perturbation = [
#             to_var(torch.from_numpy(p_), requires_grad=True, device=device)
#             for p_ in grad_accumulator
#         ]

#         # Compute hidden using perturbed past
#         perturbed_past = list(map(add, past, curr_perturbation))
#         _, _, _, curr_length, _ = curr_perturbation[0].shape

#         all_outputs = model(last, past_key_values=perturbed_past)
#         all_logits = all_outputs.logits
#         hidden = all_outputs.hidden_states[-1]
#         new_accumulated_hidden = accumulated_hidden + torch.sum(
#             hidden,
#             dim=1
#         ).detach()
#         # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
#         logits = all_logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)

#         # Compute the loss
#         loss = 0.0

#         # Compute the BoW loss
#         if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
#             for one_hot_bow in one_hot_bows_vectors:
#                 bow_logits = torch.mm(probs, torch.t(one_hot_bow))
#                 bow_loss = -torch.log(torch.sum(bow_logits))
#                 loss += bow_loss
#             if verbosity_level >= VERY_VERBOSE:
#                 print(" pplm_bow_loss:", loss.data.cpu().numpy())

#         # Compute the discriminator loss
#         if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
#             ce_loss = torch.nn.CrossEntropyLoss()
#             # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
#             curr_unpert_past = unpert_past
#             curr_probs = torch.unsqueeze(probs, dim=1)
#             wte = model.resize_token_embeddings()
#             for _ in range(horizon_length):
#                 inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
#                 _, curr_unpert_past, curr_all_hidden = model(
#                     past=curr_unpert_past,
#                     inputs_embeds=inputs_embeds
#                 )
#                 curr_hidden = curr_all_hidden[-1]
#                 new_accumulated_hidden = new_accumulated_hidden + torch.sum(
#                     curr_hidden, dim=1)

#             prediction = classifier(new_accumulated_hidden /
#                                     (curr_length + 1 + horizon_length))

#             label = torch.tensor(prediction.shape[0] * [class_label],
#                                  device=device,
#                                  dtype=torch.long)
#             discrim_loss = ce_loss(prediction, label)
#             if verbosity_level >= VERY_VERBOSE:
#                 print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
#             loss += discrim_loss

#         # Compute the KL-divergence loss
#         kl_loss = 0.0
#         if kl_scale > 0.0:
#             unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
#             unpert_probs = (
#                 unpert_probs + SMALL_CONST *
#                 (unpert_probs <= SMALL_CONST).float().to(device).detach()
#             )
#             correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
#                 device).detach()
#             corrected_probs = probs + correction.detach()
#             kl_loss = kl_scale * (
#                 (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
#             )
#             if verbosity_level >= VERY_VERBOSE:
#                 print(" kl_loss", kl_loss.data.cpu().numpy())
#             loss += kl_loss

#         loss_per_iter.append(loss.data.cpu().numpy())
#         if verbosity_level >= VERBOSE:
#             print(" pplm_loss", (loss - kl_loss).data.cpu().numpy())

#         # compute gradients
#         loss.backward()

#         # calculate gradient norms
#         if grad_norms is not None and loss_type == PPLM_BOW:
#             grad_norms = [
#                 torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
#                 for index, p_ in enumerate(curr_perturbation)
#             ]
#         else:
#             grad_norms = [
#                 (torch.norm(p_.grad * window_mask) + SMALL_CONST)
#                 for index, p_ in enumerate(curr_perturbation)
#             ]

#         # normalize gradients
#         # TODO: original PPLM code with gradient scaling
#         # grad = [
#         #     -stepsize *
#         #     (p_.grad * window_mask / grad_norms[
#         #         index] ** gamma).data.cpu().numpy()
#         #     for index, p_ in enumerate(curr_perturbation)
#         # ]
#         grad = [
#             -stepsize * p_.grad.data.cpu().numpy()
#             for p_ in curr_perturbation
#         ]

#         # accumulate gradient
#         grad_accumulator = list(map(add, grad, grad_accumulator))

#         # reset gradients, just to make sure
#         for p_ in curr_perturbation:
#             p_.grad.data.zero_()

#         # removing past from the graph
#         new_past = []
#         for p_ in past:
#             new_past.append(p_.detach())
#         past = new_past

#     # apply the accumulated perturbations to the past
#     grad_accumulator = [
#         to_var(torch.from_numpy(p_), requires_grad=True, device=device)
#         for p_ in grad_accumulator
#     ]
#     pert_past = list(map(add, past, grad_accumulator))

#     pert_past = [(attn_block[0], attn_block[1]) for attn_block in pert_past]

#     return pert_past, grad_norms, loss_per_iter


def get_classifier(
        name: Optional[str],
        class_label: Union[str, int],
        device: str,
        verbosity_level: int = REGULAR
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params["class_size"],
        embed_size=params["embed_size"]
    ).to(device)
    if "url" in params:
        state_dict = torch.hub.load_state_dict_from_url(
            params["url"], map_location=device)
    elif "path" in params:
        state_dict = torch.load(params["path"], map_location=device)
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(state_dict)
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> \
        List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            bow_url = BAG_OF_WORDS_ARCHIVE_MAP[id_or_path]
            bow_fname = BAG_OF_WORDS_ARCHIVE_MAP[id_or_path].split("/")[-1]
            bow_path = f"data/bow/{bow_fname}"
            torch.hub.download_url_to_file(bow_url, bow_path)
            filepath = bow_path
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode(word.strip(),
                              add_prefix_space=True,
                              add_special_tokens=False)
             for word in words])
    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device="cuda"):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    # bow_indices can contain multiple BoW lists, iterate through each one
    for single_bow in bow_indices:
        # Filter only words which are encoded as a max. 1 token
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        # Construct a one-hot matrix of dimensions BxV.
        # Each row is a sparse vector representation of one of the words in the BoW
        # There are vocab. size columns, for each row only one column is equal
        # to 1.
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors


def full_text_generation(
        model,
        tokenizer,
        context,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
):
    # Load the classifier
    classifier, class_id = get_classifier(
        discrim,
        class_label,
        device
    )

    # Load the BoW model.
    # bow_indices is a list of token IDs which represent the words in the BoW
    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)

    if bag_of_words and classifier:
        loss_type = PPLM_BOW_DISCRIM
        if verbosity_level >= REGULAR:
            print("Both PPLM-BoW and PPLM-Discrim are on. "
                  "This is not optimized.")

    elif bag_of_words:
        loss_type = PPLM_BOW
        if verbosity_level >= REGULAR:
            print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        if verbosity_level >= REGULAR:
            print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []

    for _ in range(num_samples):
        pert_gen_tok_text, _ = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return pert_gen_tok_texts


def generate_text_pplm(
        model,
        tokenizer,
        context,
        device="cuda",
        bow_indices=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR
):
    context_t = torch.tensor(context, device=device, dtype=torch.long)
    while len(context_t.shape) < 2:
        context_t = context_t.unsqueeze(0)
    output_so_far = context_t

    # collect one hot vectors for bags of words
    # one_hot_bows_vectors is a list of matrices. Each matrix is the one-hot
    # encoded words withing a specific BoW
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer,
                                                      device)

    grad_norms = None
    loss_in_time = []

    if verbosity_level >= QUIET:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)

    # Run the PPLM iteration. This loop performs backward propagation and then
    # forward to sample a single token.

    # Obtain the last token from the contet and compute the initial past key-values
    last = output_so_far[:, -1:]
    past = model(output_so_far[:, :-1]).past_key_values
    kl_penalty = 0.001

    for i in range_func:
        # Start with a forward pass to obtain values from the unperturbed model
        with torch.no_grad():
            unpert_outputs = model(output_so_far)
        unpert_logits = unpert_outputs.logits
        unpert_past = unpert_outputs.past_key_values
        unpert_last_hidden = unpert_outputs.hidden_states[-1]

        # Check if we are above grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # Sum the hidden representations of the last layer.
        # These are used for the classifier discriminator
        accumulated_hidden = unpert_last_hidden[:, :-1, :]
        accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

        # Perform perturbations on the past keys and values H_t
        pert_past, grad_norms, loss_this_iter, kl_penalty = perturb_past(
            past,
            model,
            last,
            unpert_past=unpert_past,
            unpert_logits=unpert_logits,
            accumulated_hidden=accumulated_hidden,
            grad_norms=grad_norms,
            stepsize=current_stepsize,
            one_hot_bows_vectors=one_hot_bows_vectors,
            classifier=classifier,
            class_label=class_label,
            loss_type=loss_type,
            num_iterations=num_iterations,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            kl_scale=kl_scale,
            device=device,
            verbosity_level=verbosity_level,
            kl_penalty=kl_penalty,
        )
        loss_in_time.append(loss_this_iter)

        # Generate model outputs using the perturbed past
        with torch.no_grad():
            pert_outputs = model(last, past_key_values=pert_past)
        pert_logits = pert_outputs.logits[:, -1, :] / temperature
        pert_probs = F.softmax(pert_logits, dim=-1)
        past = pert_outputs.past_key_values

        # TODO commented out because this is the PostNorm fusion at the end
        # # Fuse the modified model and original model
        # if perturb:

        #     unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

        #     pert_probs = ((pert_probs ** gm_scale) * (
        #         unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
        #     pert_probs = top_k_filter(pert_probs, k=top_k,
        #                               probs=True)  # + SMALL_CONST

        #     # rescale
        #     if torch.sum(pert_probs) <= 1:
        #         pert_probs = pert_probs / torch.sum(pert_probs)

        # else:
        #     pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
        #     pert_probs = F.softmax(pert_logits, dim=-1)

        # Using the perturbed model outputs determine what the next token in
        # the sqeuence will be

        pert_logits = top_k_filter(pert_logits, k=top_k)
        pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)
        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = torch.cat((output_so_far, last), dim=1)
        if verbosity_level >= REGULAR:
            print(tokenizer.decode(output_so_far.tolist()[0]))

    return output_so_far, loss_in_time


def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_weights need to be specified')
    if discrim_meta is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_meta need to be specified')

    with open(discrim_meta, "r") as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta["path"] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS["generic"] = meta


def run_pplm_example(
        cond_text: str,
        pretrained_model: str = "gpt2-medium",
        num_samples=1,
        bag_of_words=None,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        no_cuda=False,
        colorama=False,
        verbosity='regular'
):
    # set Random seed
    # TODO disable random seed in order to generate more diverse samples
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # set verbosity
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim == "generic":
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                print("discrim = {}, pretrained_model set "
                      "to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # Tokenize conditioning text
    tokenized_cond_text = tokenizer.encode(
        tokenizer.bos_token + cond_text, add_special_tokens=False)

    # Generate samples

    pert_gen_tok_texts = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        discrim=discrim,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity_level=verbosity_level
    )

    with open("results.txt", "w") as f:
        f.write("\n".join([tokenizer.decode(pert_gen_tok_text.tolist()[0]).replace(
            "\n", "") for pert_gen_tok_text in pert_gen_tok_texts]))

    # The code below prints the perturbed text making sure relevant words are
    # coloured

    # bow_word_ids = set()
    # if bag_of_words and colorama:
    #     bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
    #                                            tokenizer)
    #     for single_bow_list in bow_indices:
    #         # filtering all words in the list composed of more than 1 token
    #         filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
    #         # w[0] because we are sure w has only 1 item because previous fitler
    #         bow_word_ids.update(w[0] for w in filtered)

    # # iterate through the perturbed texts
    # for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
    #     try:
    #         # untokenize unperturbed text
    #         if colorama:
    #             import colorama

    #             pert_gen_text = ""
    #             for word_id in pert_gen_tok_text.tolist()[0]:
    #                 if word_id in bow_word_ids:
    #                     pert_gen_text += '{}{}{}'.format(
    #                         colorama.Fore.RED,
    #                         tokenizer.decode([word_id]),
    #                         colorama.Style.RESET_ALL
    #                     )
    #                 else:
    #                     pert_gen_text += tokenizer.decode([word_id])
    #         else:
    #             pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])

    #         print("= Perturbed generated text {} =".format(i + 1))
    #         print(pert_gen_text)
    #         print()
    #     except:
    #         pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, required=True,
        help="Prefix texts to condition on"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default="military",
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None,
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None,
                        help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")

    args = parser.parse_args()
    run_pplm_example(**vars(args))
