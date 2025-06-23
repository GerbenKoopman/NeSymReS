import torch
import torch.nn.functional as F
from typing import List, Tuple


def generate_beam(
    env,
    dec,
    decoder_args,
    beam_size: int,
    length_penalty: float,
    early_stopping: bool,
    max_len: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, List["BeamHypotheses"]]:
    """
    Decode a sentence given initial start using beam search.

    Args:
        env: Environment containing vocabulary and special tokens.
        dec: Decoder model.
        decoder_args: Tuple containing decoder arguments (target, encoder source, masks).
        beam_size: Number of beams for beam search.
        length_penalty: Penalty for longer sequences.
        early_stopping: Whether to stop early when conditions are met.
        max_len: Maximum length of the generated sequence.

    Returns:
        Tuple containing decoded tensor, target lengths, and generated hypotheses.
    """
    trg, enc_src, trg_mask, src_mask = decoder_args
    src_enc = enc_src

    assert beam_size >= 1

    bs = len(src_enc)
    n_words = env.n_words

    src_enc = expand_to_beam_size(src_enc, bs, beam_size)

    generated, positions = initialize_generated_tensors(
        env, src_enc, max_len, bs, beam_size
    )
    generated_hyps = [
        BeamHypotheses(beam_size, max_len, length_penalty, early_stopping)
        for _ in range(bs)
    ]

    beam_scores = initialize_beam_scores(src_enc, bs, beam_size)
    cache = {"slen": 0}
    done = [False for _ in range(bs)]
    cur_len = 1

    while cur_len < max_len:
        scores = compute_word_scores(
            dec,
            generated,
            cur_len,
            positions,
            src_enc,
            src_mask,
            cache,
            bs,
            beam_size,
            env,
        )
        next_batch_beam = select_next_beam(
            scores,
            beam_scores,
            bs,
            beam_size,
            n_words,
            generated_hyps,
            done,
            cur_len,
            max_len,
            env,
            generated,
        )

        beam_scores, beam_words, beam_idx = process_next_beam(
            next_batch_beam, src_enc, bs, beam_size
        )
        generated, cache = reorder_states(
            generated, cache, beam_idx, cur_len, beam_words
        )

        cur_len += 1
        if all(done):
            break

    return finalize_hypotheses(generated_hyps, src_enc, bs, env, max_len)


def expand_to_beam_size(src_enc, bs: int, beam_size: int):
    return (
        src_enc.unsqueeze(1)
        .expand((bs, beam_size) + src_enc.shape[1:])
        .contiguous()
        .view((bs * beam_size,) + src_enc.shape[1:])
    )


def initialize_generated_tensors(env, src_enc, max_len: int, bs: int, beam_size: int):
    generated = src_enc.new(max_len, bs * beam_size).fill_(env.pad_index)
    generated[0].fill_(env.eos_index)
    positions = (
        torch.arange(max_len, device=src_enc.device).unsqueeze(1).expand_as(generated)
    )
    return generated, positions


def initialize_beam_scores(src_enc, bs: int, beam_size: int):
    beam_scores = src_enc.new(bs, beam_size).fill_(0)
    beam_scores[:, 1:] = -1e9
    return beam_scores.view(-1)


def compute_word_scores(
    dec,
    generated,
    cur_len: int,
    positions,
    src_enc,
    src_mask,
    cache,
    bs: int,
    beam_size: int,
    env,
):
    tensor = dec(
        x=generated[:cur_len],
        lengths=src_enc.new(bs * beam_size).fill_(cur_len),
        positions=positions[:cur_len],
        causal=True,
        src_enc=src_enc,
        src_len=src_mask,
        cache=cache,
    )
    tensor = tensor.data[-1, :, :]
    scores = env.proj(tensor)
    return F.log_softmax(scores, dim=-1)


def select_next_beam(
    scores,
    beam_scores,
    bs: int,
    beam_size: int,
    n_words: int,
    generated_hyps,
    done,
    cur_len: int,
    max_len: int,
    env,
    generated,
):
    _scores = scores + beam_scores[:, None].expand_as(scores)
    _scores = _scores.view(bs, beam_size * n_words)

    next_scores, next_words = torch.topk(
        _scores, 2 * beam_size, dim=1, largest=True, sorted=True
    )
    next_batch_beam = []

    for sent_id in range(bs):
        if done[sent_id]:
            next_batch_beam.extend([(0, env.pad_index, 0)] * beam_size)
            continue

        next_sent_beam = []
        for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
            beam_id = idx // n_words
            word_id = idx % n_words

            if word_id == env.eos_index or cur_len + 1 == max_len:
                generated_hyps[sent_id].add(
                    generated[:cur_len, sent_id * beam_size + beam_id].clone().cpu(),
                    value.item(),
                )
            else:
                next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

            if len(next_sent_beam) == beam_size:
                break

        if len(next_sent_beam) == 0:
            next_sent_beam = [(0, env.pad_index, 0)] * beam_size
        next_batch_beam.extend(next_sent_beam)

    return next_batch_beam


def process_next_beam(next_batch_beam, src_enc, bs: int, beam_size: int):
    beam_scores = src_enc.new([x[0] for x in next_batch_beam])
    beam_words = src_enc.new([x[1] for x in next_batch_beam])
    beam_idx = src_enc.new([x[2] for x in next_batch_beam])
    return beam_scores, beam_words, beam_idx


def reorder_states(generated, cache, beam_idx, cur_len: int, beam_words):
    generated = generated[:, beam_idx]
    generated[cur_len] = beam_words
    for k in cache.keys():
        if k != "slen" and isinstance(cache[k], (tuple, list)):
            cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])
    return generated, cache


def finalize_hypotheses(generated_hyps, src_enc, bs: int, env, max_len: int):
    tgt_len = src_enc.new(bs)
    best = []

    for i, hypotheses in enumerate(generated_hyps):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        tgt_len[i] = len(best_hyp) + 1
        best.append(best_hyp)

    decoded = src_enc.new(tgt_len.max().item(), bs).fill_(env.pad_index)
    for i, hypo in enumerate(best):
        decoded[: tgt_len[i] - 1, i] = hypo
        decoded[tgt_len[i] - 1, i] = env.eos_index

    return decoded, tgt_len, generated_hyps


class BeamHypotheses:
    def __init__(
        self, n_hyp: int, max_len: int, length_penalty: float, early_stopping: bool
    ):
        self.max_len = max_len - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.hyp)

    def add(self, hyp: torch.Tensor, sum_logprobs: float):
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted((s, idx) for idx, (s, _) in enumerate(self.hyp))
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float):
        if len(self) < self.n_hyp:
            return False
        if self.early_stopping:
            return True
        return self.worst_score >= best_sum_logprobs / self.max_len**self.length_penalty


def generate_multiple_beams(
    env,
    dec,
    decoder_args,
    beam_configs: List[dict],
    early_stopping: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor, List["BeamHypotheses"]]]:
    """
    Run multiple beam search configurations and return predictions for each.

    Args:
        env: Environment containing vocabulary and special tokens.
        dec: Decoder model.
        decoder_args: Tuple containing decoder arguments (target, encoder source, masks).
        beam_configs: List of configurations for beam search (beam_size, length_penalty, max_len).
        early_stopping: Whether to stop early when conditions are met.

    Returns:
        List of tuples containing decoded tensor, target lengths, and generated hypotheses for each configuration.
    """
    results = []
    for config in beam_configs:
        beam_size = config.get("beam_size", 5)
        length_penalty = config.get("length_penalty", 1.0)
        max_len = config.get("max_len", 100)

        result = generate_beam(
            env,
            dec,
            decoder_args,
            beam_size,
            length_penalty,
            early_stopping,
            max_len,
        )
        results.append(result)

    return results
