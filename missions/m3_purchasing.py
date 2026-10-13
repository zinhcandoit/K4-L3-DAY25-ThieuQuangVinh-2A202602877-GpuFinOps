"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5: Carbon-aware scheduling for interruptible workloads
    from finops import sustainability
    interruptible_jobs = [j for j in jobs if bool(int(num(j["interruptible"])))]
    total_kwh_month = sum(
        (num(j["hours_per_day"]) * DAYS * int(num(j["num_gpus"])) * num(cat[j["gpu_type"]]["watts"])) / 1000.0
        for j in interruptible_jobs
    )
    carbon_us_east = (total_kwh_month * sustainability.REGION_CARBON["us-east-1"]) / 1000.0
    carbon_norway = (total_kwh_month * sustainability.REGION_CARBON["europe-north1"]) / 1000.0
    co2_saved_kg = carbon_us_east - carbon_norway
    co2_saved_pct = (co2_saved_kg / carbon_us_east * 100.0) if carbon_us_east else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print("\n-- [Extension 5] Carbon-Aware Scheduling (Interruptible Jobs) --")
        print(f"Total interruptible compute energy: {total_kwh_month:,.1f} kWh/month across {len(interruptible_jobs)} jobs")
        print(f"Baseline carbon (us-east-1): {carbon_us_east:,.2f} kg CO2e")
        print(f"Cleanest carbon (europe-north1): {carbon_norway:,.2f} kg CO2e")
        print(f"Carbon reduction: {co2_saved_kg:,.2f} kg CO2e ({co2_saved_pct:.1f}% saved)")
        print("\nRegional Comparison for Interruptible Fleet:")
        print(f"{'Region':18}{'gCO2/kWh':>10}{'$/kWh':>8}{'kg CO2e':>12}{'Power ($)':>12}")
        for reg, gco2 in sustainability.REGION_CARBON.items():
            pkwh = sustainability.REGION_PRICE_KWH[reg]
            co2_kg = (total_kwh_month * gco2) / 1000.0
            p_usd = total_kwh_month * pkwh
            print(f"{reg:18}{gco2:>10}{pkwh:>8.3f}{co2_kg:>12.2f}${p_usd:>11.2f}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "extension_carbon": {
                "total_kwh_month": round(total_kwh_month, 1),
                "carbon_us_east_kg": round(carbon_us_east, 2),
                "carbon_norway_kg": round(carbon_norway, 2),
                "co2_saved_kg": round(co2_saved_kg, 2),
                "co2_saved_pct": round(co2_saved_pct, 1),
            }}


if __name__ == "__main__":
    run()
