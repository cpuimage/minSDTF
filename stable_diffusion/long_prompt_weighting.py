import re
from typing import List, Optional, Union

import numpy as np

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(tokenizer, prompt: List[str], max_length: int, embedding_tokens_count: int = 0,
                             embedding_tokens_weight: float = 1.0):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        if embedding_tokens_count > 0:
            # add embedding_tokens for init
            embedding_tokens = tokenizer.encode("*")[1:-1]
            text_token += embedding_tokens * embedding_tokens_count
            text_weight += [embedding_tokens_weight] * embedding_tokens_count
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = tokenizer.encode(word.strip())[1:-1]
            text_token += list(token)
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        print("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [pad] * (max_length - 1 - len(tokens[i]) - 1) + [eos]
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2): min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
        text_clip_embedding,
        text_encoder,
        text_input: np.array,
        chunk_length: int,
        no_boseos_middle: Optional[bool] = True,
        embedding_tokens_count=0,
        embedding=None,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    use_embedding = embedding_tokens_count > 0 and embedding is not None
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2): (i + 1) * (chunk_length - 2) + 2].copy()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            if use_embedding and i == 0:
                clip_embedding = text_clip_embedding.predict_on_batch(
                    [text_input_chunk, np.asarray([list(range(text_input_chunk.shape[1]))], dtype=np.int32)])
                clip_embedding = np.concatenate(
                    [clip_embedding[:, 0:1, :],
                     np.tile(embedding, (clip_embedding.shape[0], 1, 1)).astype(clip_embedding.dtype),
                     clip_embedding[:, embedding_tokens_count + 1:, :]], axis=1)
                text_embedding = text_encoder.predict_on_batch(clip_embedding)
            else:
                clip_embedding = text_clip_embedding.predict_on_batch(
                    [text_input_chunk, np.asarray([list(range(text_input_chunk.shape[1]))], dtype=np.int32)])
                text_embedding = text_encoder.predict_on_batch(clip_embedding)

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = np.concatenate(text_embeddings, axis=1)
    else:
        clip_embedding = text_clip_embedding.predict_on_batch(
            [text_input, np.asarray([list(range(text_input.shape[1]))], dtype=np.int32)])
        if use_embedding:
            clip_embedding = np.concatenate(
                [clip_embedding[:, 0:1, :],
                 np.tile(embedding, (clip_embedding.shape[0], 1, 1)).astype(clip_embedding.dtype),
                 clip_embedding[:, embedding_tokens_count + 1:, :]], axis=1)
        text_embeddings = text_encoder.predict_on_batch(clip_embedding)
    return text_embeddings


def get_weighted_text_embeddings(
        tokenizer,
        text_clip_embedding,
        text_encoder,
        prompt: Union[str, List[str]],
        max_embeddings_multiples: Optional[int] = 4,
        no_boseos_middle: Optional[bool] = False,
        skip_parsing: Optional[bool] = False,
        skip_weighting: Optional[bool] = False,
        model_max_length=77,
        pad_token_id=49407,
        embedding_tokens_count=0,
        embedding_tokens_weight=1.0,
        embedding=None,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        tokenizer  : provide access to the tokenizer
        text_encoder :  provide access to the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        max_embeddings_multiples (`int`, *optional*, defaults to `1`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    if embedding_tokens_count > 0 and embedding is None:
        embedding_tokens_count = 0
    max_length = (model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(tokenizer, prompt, max_length - 2,
                                                                 embedding_tokens_count, embedding_tokens_weight)
    else:
        prompt_tokens = [
            token[1:-1]
            for token in tokenizer.encode(prompt)[:max_length]
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = tokenizer.start_of_text
    eos = tokenizer.end_of_text
    pad = pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=model_max_length,
    )
    prompt_tokens = np.array(prompt_tokens, dtype=np.int32)
    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        text_clip_embedding,
        text_encoder,
        prompt_tokens,
        model_max_length,
        no_boseos_middle=no_boseos_middle,
        embedding_tokens_count=embedding_tokens_count,
        embedding=embedding,
    )
    prompt_weights = np.array(prompt_weights, dtype=text_embeddings.dtype)
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.mean(axis=(-2, -1))
        text_embeddings *= prompt_weights[:, :, None]
        text_embeddings *= (previous_mean / text_embeddings.mean(axis=(-2, -1)))[:, None, None]
    return text_embeddings
