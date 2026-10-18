"""Bounded centralized CoopHE answers: start in <auxiliary>, stop at </main>.

Open joint answers dropped the section tags after a few updates and kept
writing prose after the code until the token budget ran out, and an answer
cut off before its functions are complete scores zero. With
``magrpo.centralized_answer=bounded`` the prompt ends with the opening
<auxiliary> tag and every train and eval rollout stops at the first </main>.
The reward, the parser and the decentralized runs are unchanged.
"""

import re
from typing import Any, Dict, List, Optional, Sequence

from comlrl.trainers.reinforce import CentralizedMAGRPOTrainer

AUXILIARY_PREFILL = "<auxiliary>\n"
MAIN_STOP = "</main>"


class AuxiliaryPrefillAdapter:
    """Wrap a CoopHE adapter so the joint answer starts inside <auxiliary>."""

    prefill = AUXILIARY_PREFILL

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def build_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
    ) -> str:
        return self.adapter.build_prompt(batch_item, agent_prompts) + self.prefill

    def parse_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        num_agents: int,
    ) -> Sequence[str]:
        # The opening tag belongs to the prompt; restore it before parsing.
        if not re.match(r"\s*<\s*auxiliary\s*>", completion, flags=re.IGNORECASE):
            completion = self.prefill + completion
        return self.adapter.parse_completion(completion, batch_item, num_agents)


def crop_stop_padding(result: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    """Keep exactly the sampled tokens of rollouts that stopped at </main>.

    Generation pads a sequence once it reaches the stop string. When the pad
    token is also EOS, CoMLRL counts the first pad as a sampled EOS and scores
    it in the joint loss, although the stop criterion forced it. Rollouts that
    ended with a real EOS or at the token budget are left unchanged.
    """
    pad_id = tokenizer.pad_token_id
    tokens: List[Any] = []
    lengths: List[int] = []
    for sequence, length in zip(result["completion_input_ids"][0], result["response_lens"]):
        if pad_id is not None:
            pads = (sequence == pad_id).nonzero()
            if pads.numel():
                end = int(pads[0].item())
                text = tokenizer.decode(sequence[:end], skip_special_tokens=True)
                if MAIN_STOP in text:
                    sequence, length = sequence[:end], end
        tokens.append(sequence)
        lengths.append(int(length))
    cropped = dict(result)
    cropped["completion_input_ids"] = [tokens]
    cropped["completion_attention_mask"] = [
        [
            mask[: len(sequence)]
            for mask, sequence in zip(result["completion_attention_mask"][0], tokens)
        ]
    ]
    cropped["response_lens"] = lengths
    return cropped


class StopAfterMainMAGRPOTrainer(CentralizedMAGRPOTrainer):
    """Centralized MAGRPO whose train and eval rollouts stop at </main>."""

    def _generate_completions(
        self,
        agent: Any,
        batch_items: List[Dict[str, Any]],
        agent_idx: int = 0,
        num_return_sequences: int = 1,
        max_new_tokens: int = 128,
        prompts_override: Optional[List[str]] = None,
        external_prompts: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tokenizer = self.tokenizers[agent_idx]
        kwargs.setdefault("stop_strings", [MAIN_STOP])
        kwargs.setdefault("tokenizer", tokenizer)
        result = super()._generate_completions(
            agent,
            batch_items,
            agent_idx=agent_idx,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            prompts_override=prompts_override,
            external_prompts=external_prompts,
            **kwargs,
        )
        return crop_stop_padding(result, tokenizer)
