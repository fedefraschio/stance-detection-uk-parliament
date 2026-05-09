import json
import random
import re
import numpy as np
import ollama
from scipy.stats import spearmanr, kendalltau


class Evaluator:
    def generate_gold_standard(self, parties: list, anchors: dict, years: list,
                               model="qwen3:8b", random_seed=42, debug_mode=False) -> list:
        parties = parties.copy()
        random.shuffle(parties)

        parties_json = json.dumps(parties)
        example = json.dumps(parties[::-1], ensure_ascii=False)

        prompt = f"""You are a political scientist specialising in UK parliamentary politics ({years[0]}–{years[-1]}).

            Rank these parties from most aligned with POSITION_A to most aligned with POSITION_B:

            ISSUE: {anchors['topic']}
            POSITION_A: {anchors['con']}
            POSITION_B: {anchors['pro']}

            Parties to rank (use these exact strings):
            {parties_json}

            Return ONLY a JSON array ordered from most POSITION_A-aligned to most POSITION_B-aligned.
            Example format (NOT the correct answer):
            {example}

            OUTPUT:"""

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            think=True,
            options={"temperature": 0.25, "seed": random_seed}
        )

        content = response["message"]["content"].strip()

        match = re.search(r"\[[^\]]+\]", content)
        if not match:
            raise ValueError(f"No JSON array found in response:\n{content}")

        stripped_array = match.group().strip("\n")

        if stripped_array.count("[") != 1 or stripped_array.count("]") != 1:
            raise ValueError(f"Invalid JSON array format:\n{stripped_array}")

        if debug_mode:
            print("Parties to rank:", parties)
            print("Raw LLM response:", content)

        ranked_parties = json.loads(stripped_array)

        if set(ranked_parties) != set(parties):
            raise ValueError(f"Party mismatch.\nExpected: {parties}\nGot: {ranked_parties}\nRaw: {content}")

        if len(ranked_parties) != len(parties):
            raise ValueError(f"Duplicate or missing parties.\nExpected {len(parties)} got {len(ranked_parties)}")

        return ranked_parties

    def evaluate_ordering(self, pred_ordering: list, gold_ordering: list) -> dict:
        common = [p for p in pred_ordering if p in gold_ordering]
        n = len(common)

        if n < 2:
            print(f"Warning: only {n} party/parties in common — metrics not computable.")
            return {
                "spearman_rho": np.nan, "spearman_p": np.nan,
                "kendall_tau": np.nan, "kendall_p": np.nan,
                "lcs_ratio": np.nan, "n_parties": n
            }

        pred_ranks = [pred_ordering.index(p) for p in common]
        gold_ranks = [gold_ordering.index(p) for p in common]

        rho, p_rho = spearmanr(pred_ranks, gold_ranks)
        tau, p_tau = kendalltau(pred_ranks, gold_ranks)

        gold_common = [p for p in gold_ordering if p in common]
        m, n_gold = len(common), len(gold_common)
        dp = [[0] * (n_gold + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n_gold + 1):
                if common[i-1] == gold_common[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n_gold] / max(m, n_gold)

        return {
            "spearman_rho": round(rho, 4),
            "spearman_p": round(p_rho, 4),
            "kendall_tau": round(tau, 4),
            "kendall_p": round(p_tau, 4),
            "lcs_ratio": round(lcs, 4),
            "n_parties": n
        }
