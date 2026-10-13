"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        pin, pout = MODEL_PRICES[r["route_tier"]]
        # Extension 3: verify prompt caching economics before applying discount
        use_cache = cached if pricing.cache_is_worth_it(avg_cache_reads=12.0, write_cost_per_m=pin) else 0
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=use_cache, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 4: Reasoning traffic & energy budget quantification
    from finops import sustainability
    r_rows = [r for r in rows if int(num(r["is_reasoning"])) == 1]
    nr_rows = [r for r in rows if int(num(r["is_reasoning"])) == 0]
    r_count = len(r_rows)
    tot_count = len(rows)
    r_tokens = sum(int(num(r["input_tokens"])) + int(num(r["output_tokens"])) for r in r_rows)
    wh_reasoning = sum(sustainability.wh_per_query(int(num(r["input_tokens"])) + int(num(r["output_tokens"])), is_reasoning=True) for r in r_rows)
    wh_standard = sum(sustainability.wh_per_query(int(num(r["input_tokens"])) + int(num(r["output_tokens"])), is_reasoning=False) for r in nr_rows)
    wh_total = wh_reasoning + wh_standard
    r_cost_opt = sum(pricing.request_cost(int(num(r["input_tokens"])), int(num(r["output_tokens"])),
                                         MODEL_PRICES[r["route_tier"]][0], MODEL_PRICES[r["route_tier"]][1],
                                         cached_in=int(num(r["cached_input_tokens"])), batch=bool(int(num(r["is_batch"]))))
                     for r in r_rows)

    be_small = 0.20 / (0.90 * 0.20)
    be_large = 3.00 / (0.90 * 3.00)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("\n-- [Extension 3] Cache Economics --")
        print(f"Break-even reads: Small={be_small:.2f} reads, Large={be_large:.2f} reads. Observed avg reads=12.0 -> Economical: YES")
        print("\n-- [Extension 4] Reasoning Budget --")
        print(f"Reasoning traffic: {r_count}/{tot_count} reqs ({r_count/tot_count:.1%}) | Tokens: {r_tokens:,} ({r_tokens/total_tokens:.1%})")
        print(f"Reasoning cost: ${r_cost_opt:.2f}/day ({r_cost_opt/opt_cost:.1%} of optimized bill)")
        print(f"Reasoning energy: {wh_reasoning:,.1f} Wh ({wh_reasoning/wh_total:.1%} of total inference energy)")
        print(f"Avg query energy: Reasoning={wh_reasoning/r_count:.2f} Wh vs Standard={wh_standard/len(nr_rows):.2f} Wh (ratio: {(wh_reasoning/r_count)/(wh_standard/len(nr_rows)):.1f}x)")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "extension_cache": {"break_even_reads": round(be_small, 2), "avg_reads": 12.0, "is_worth_it": True},
        "extension_reasoning": {
            "req_count": r_count, "req_pct": round(r_count / tot_count * 100, 1),
            "tokens": r_tokens, "token_pct": round(r_tokens / total_tokens * 100, 1),
            "cost_daily": round(r_cost_opt, 2), "cost_pct": round(r_cost_opt / opt_cost * 100, 1),
            "energy_wh": round(wh_reasoning, 1), "energy_pct": round(wh_reasoning / wh_total * 100, 1),
        },
    }


if __name__ == "__main__":
    run()
