import random
import re
import time
from typing import List, Optional, Tuple

import affine as af

static = ("In the following, we will test your ability to understand and execute python code.\n\n"
          "RULES:\n"
          "* You will not be able to use any external tools, such as a calculator or python during this test\n"
          "* You will see one python snippet per message, with every snippet building on top of the previous state\n"
          "* You may think and decypher the test for as long as you want\n"
          "* Return the *exact* printout the python interpreter would otherwise return, including formatting and potential typos in the challenge\n"
          "* Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated\n"
          "* You may write any text outside of the <ANSWER> tags\n\n"
          "Example:\n"
          "CHALLENGE:\n"
          ">>> var1 = [62139]\n"
          ">>> var2 = [62598]\n"
          ">>> print(var1)\n"
          ">>> var4 = [40899]\n"
          ">>> var5 = sorted(var2)\n"
          ">>> var7 = 86093\n"
          ">>> var7 += 35501\n"
          ">>> print(var7)\n"
          "RESPONSE:\n"
          "Okay, I've processed the code you provided.\n<ANSWER>[62139]\n121594\n</ANSWER>\n\n"
          "Below, you will see the real task. Remember and follow the rules.\n\n"
          "CHALLENGE:\n")


def _generate_lists(target_length: int, turns: int, ops_per_turn: int, max_digits: int, seed: Optional[int] = None) -> \
        Tuple[List[str], List[str]]:
    rng = random.Random(time.time_ns() if seed is None else seed)  # can't use "or" syntax as we may want a seed of 0
    lists = [[rng.randint(0, 10 ** max_digits - 1) for _ in range(target_length)]]
    ops = [f'>>> x = {lists[0]}']
    for _ in range(turns):
        current = lists[-1][:]
        op = []
        for _ in range(ops_per_turn):
            if rng.random() < 0.5:
                del_at = rng.randint(0, len(current) - 1)
                op.append(f"x.remove({current[del_at]})")
                del current[del_at]
            else:
                val = rng.randint(0, 10 ** max_digits - 1)
                idx = rng.randint(0, len(current))
                op.append(f"x.insert({idx}, {val})")
                current.insert(idx, val)
        lists.append(sorted(current))
        ops.append('\n'.join(f'>>> {x}' for x in op))

    return [str(x) + '\n>>> print(sorted(x))' for x in lists], ops


class MTS(af.BaseEnv):
    """
    Multi-Turn Sorting with random Insertion and Deletion

    This tests how well a model can keep state using the trivial proxy of inserting into a sorted list

    The 0th turn will always present a full list and ask for it to be sorted, rather than computing any operations on it
    """
    __version__: str = "0.0.1"
    target_length: int
    turns: int
    ops_per_turn: int
    max_digits: int

    def __init__(self, target_length: int = 16, turns: int = 4, ops_per_turn: int = 4, max_digits: int = 5):
        super().__init__(target_length=target_length, turns=turns, ops_per_turn=ops_per_turn, max_digits=max_digits)

    async def generate(self):
        targets, ops = _generate_lists(self.target_length, self.turns, self.ops_per_turn, self.max_digits)
        return af.Challenge(env=self, prompt='', extra={'timestamp': time.time()},  #
                            task_sequence=[(static + o, {'target': t}) for t, o in zip(targets, ops)])

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        matches = re.findall(r"<ANSWER>(.*?)</ANSWER>", response.response or "", re.IGNORECASE | re.DOTALL)
        if not matches:
            return af.Evaluation(env=self, score=0.0)
        match = list(matches)[0].strip()
        state: af.ChallengeEvaluationState = challenge.extra["state"]
        target = challenge.extra["target"].strip()
        ok = float(target == match)
        if not ok:
            state.early_exit = True
        return af.Evaluation(env=self, score=float(ok))
