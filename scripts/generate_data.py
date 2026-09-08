"""Synthetic supply-chain dataset generator for replenish-copilot.

Stdlib only: csv, os, random, datetime (+ pathlib, collections).
Seeds realistic values from the Kaggle dump at repo root
(supply_chain_dataset1.csv, 91k rows) when present, otherwise
falls back to pure synthetic generation.

Outputs:
  data/csv/inventory.csv        (75 SKUs)
  data/csv/suppliers.csv        (15 vendors)
  data/csv/purchase_orders.csv  (150 POs)
  data/csv/demand.csv           (600 daily records)
  data/policies/POL-*.md        (6 policy docs, each with doc_id header)
"""

from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_CSV = ROOT / "supply_chain_dataset1.csv"
CSV_DIR = ROOT / "data" / "csv"
POLICY_DIR = ROOT / "data" / "policies"

SEED = 42
N_SKUS = 75
N_SUPPLIERS = 15
N_POS = 150
N_DEMAND = 600

CATEGORIES = ["Electronics", "Hardware", "Consumables", "Packaging", "Spare Parts"]
PRODUCT_ADJ = ["Pro", "Max", "Eco", "Prime", "Lite", "Ultra"]
PRODUCT_BASE = [
    "Resistor Kit", "Bearing Set", "Lubricant", "Carton Box", "Sensor Module",
    "Fastener Pack", "Cable Harness", "Filter Cartridge", "Gasket Ring", "Motor Mount",
]

STATUSES = ["delivered", "delivered", "delivered", "late", "pending"]


def norm_sku(raw: str) -> str:
    """SKU_1 -> SKU-001, SKU-014 stays SKU-014."""
    digits = "".join(ch for ch in raw if ch.isdigit())
    return f"SKU-{int(digits):03d}" if digits else raw


def norm_sup(raw: str) -> str:
    digits = "".join(ch for ch in raw if ch.isdigit())
    return f"SUP-{int(digits):02d}" if digits else raw


def load_kaggle_seed() -> dict | None:
    """Extract per-SKU snapshots + supplier lead times from Kaggle dump."""
    if not KAGGLE_CSV.exists():
        return None
    latest: dict[str, dict] = {}
    sup_lead: dict[str, list[int]] = defaultdict(list)
    demand_pool: list[dict] = []
    with open(KAGGLE_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sku = norm_sku(row["SKU_ID"])
            sup = norm_sup(row["Supplier_ID"])
            sup_lead[sup].append(int(row["Supplier_Lead_Time_Days"]))
            # keep latest row per SKU as inventory snapshot
            if sku not in latest or row["Date"] >= latest[sku]["Date"]:
                latest[sku] = row
            if len(demand_pool) < 5000 and random.random() < 0.06:
                demand_pool.append(row)
    return {"latest": latest, "sup_lead": sup_lead, "demand_pool": demand_pool}


def build_suppliers(seed: dict | None) -> list[dict]:
    suppliers: list[dict] = []
    real = sorted((seed or {}).get("sup_lead", {}).keys())
    for i in range(1, N_SUPPLIERS + 1):
        sid = f"SUP-{i:02d}"
        if i <= len(real):
            leads = seed["sup_lead"][real[i - 1]]
            lead = round(sum(leads) / len(leads))
        else:  # synthetic top-up beyond Kaggle's 10 suppliers
            lead = random.choice([2, 3, 4, 5, 7, 10, 14])
        # make 2 suppliers chronically late so SLA queries have answers
        sla = round(random.uniform(0.68, 0.82), 3) if i in (4, 11) else round(random.uniform(0.88, 0.99), 3)
        suppliers.append({
            "supplier_id": sid,
            "supplier_name": f"Vendor {sid} {'Industrial' if i % 2 else 'Supply Co.'}",
            "lead_time_days": lead,
            "sla_compliance_rate": sla,
            "contact_email": f"ops{sid.lower().replace('-', '')}@example-vendor.com",
        })
    return suppliers


def build_inventory(seed: dict | None, suppliers: list[dict]) -> list[dict]:
    inv: list[dict] = []
    latest = (seed or {}).get("latest", {})
    real_skus = sorted(latest.keys())
    sup_ids = [s["supplier_id"] for s in suppliers]
    for i in range(1, N_SKUS + 1):
        sku = f"SKU-{i:03d}"
        if i <= len(real_skus):
            row = latest[real_skus[i - 1]]
            stock = int(row["Inventory_Level"])
            rop = int(row["Reorder_Point"])
            cost = float(row["Unit_Cost"])
            sup = norm_sup(row["Supplier_ID"])
            sup = sup if sup in sup_ids else random.choice(sup_ids)
        else:  # synthetic top-up beyond Kaggle's 50 SKUs
            stock = random.randint(0, 600)
            rop = random.randint(80, 400)
            cost = round(random.uniform(2.5, 120.0), 2)
            sup = random.choice(sup_ids)
        # force a few SKUs into stockout-risk territory for eval queries
        if i in (14, 32, 7):
            stock = random.randint(0, max(5, rop // 6))
        safety = max(5, round(rop * random.uniform(0.2, 0.35)))
        inv.append({
            "sku_id": sku,
            "product_name": f"{random.choice(PRODUCT_BASE)} {random.choice(PRODUCT_ADJ)} {i}",
            "category": CATEGORIES[i % len(CATEGORIES)],
            "stock_on_hand": stock,
            "reorder_point": rop,
            "safety_stock": safety,
            "unit_cost_usd": cost,
            "supplier_id": sup,
        })
    return inv


def build_pos(inventory: list[dict]) -> list[dict]:
    pos: list[dict] = []
    start = date(2024, 10, 1)
    for i in range(1, N_POS + 1):
        item = random.choice(inventory)
        qty = random.randint(50, 1000)
        odate = start + timedelta(days=random.randint(0, 80))
        # supplier 04/11 ship late more often -> SLA violation queries work
        late_bias = item["supplier_id"] in ("SUP-04", "SUP-11")
        status = random.choices(
            ["delivered", "late", "pending"],
            weights=[0.55, 0.35, 0.10] if late_bias else [0.8, 0.1, 0.1],
        )[0]
        expected = odate + timedelta(days=random.choice([2, 3, 4, 5, 7, 14]))
        pos.append({
            "po_id": f"PO-{i:04d}",
            "sku_id": item["sku_id"],
            "supplier_id": item["supplier_id"],
            "order_qty": qty,
            "order_date": odate.isoformat(),
            "expected_delivery": expected.isoformat(),
            "status": status,
        })
    return pos


def build_demand(seed: dict | None, inventory: list[dict]) -> list[dict]:
    pool = (seed or {}).get("demand_pool", []) if seed else []
    out: list[dict] = []
    skus = [r["sku_id"] for r in inventory]
    start = date(2024, 11, 1)
    for i in range(N_DEMAND):
        if pool and random.random() < 0.7:
            row = random.choice(pool)
            sku = norm_sku(row["SKU_ID"])
            sku = sku if sku in skus else random.choice(skus)
            out.append({
                "date": row["Date"],
                "sku_id": sku,
                "daily_demand": row["Units_Sold"],
                "forecasted_demand": row["Demand_Forecast"],
            })
        else:
            sku = random.choice(skus)
            daily = random.randint(0, 60)
            out.append({
                "date": (start + timedelta(days=random.randint(0, 59))).isoformat(),
                "sku_id": sku,
                "daily_demand": daily,
                "forecasted_demand": round(daily * random.uniform(0.85, 1.2), 2),
            })
    out.sort(key=lambda r: (r["date"], r["sku_id"]))
    return out


POLICIES: list[tuple[str, str, str]] = [
    ("POL-001", "Safety Stock Formulas",
     """# Safety Stock Formulas

## Purpose
Define how safety stock and reorder points are calculated for all SKUs.

## Formulas
- Reorder Point = (Average Daily Demand x Average Lead Time) + Safety Stock
- Safety Stock = Z-score x Demand StdDev x sqrt(Avg Lead Time)
- Default service level Z = 1.65 (95%) for A-items, 1.28 (90%) for B-items, 1.04 (85%) for C-items.

## Worked example
SKU-032: avg daily demand 20, lead time 7 days, stddev 6 -> safety stock = 1.65 x 6 x sqrt(7) = ~26 units.
Reorder point = (20 x 7) + 26 = 166 units.

## Rule
Never set safety stock to zero for A-items. Planner must approve any override.
"""),
    ("POL-002", "Supplier SLA Penalties",
     """# Supplier SLA Penalties

## SLA thresholds
- Standard lead time per supplier master. Delivery after expected_delivery + 2 day grace = SLA breach.
- On-time rate below 90% triggers review; below 80% triggers penalty + sourcing review.

## Penalties
- 2% invoice credit per late PO, capped at 10% per quarter.
- Repeated breach (3 lates in 30 days) escalates to procurement lead.

## Escalation
Planner flags breach in copilot, contacts supplier ops email, opens backup PO if stock covers < 14 days.
"""),
    ("POL-003", "Stockout Prioritization",
     """# Stockout Prioritization

## Risk definition
- A SKU is at stockout risk when stock_on_hand <= reorder_point.
- Critical when stock_on_hand <= safety_stock.

## Prioritization order
1. Critical A-items (high unit cost or high velocity) first.
2. Then at-risk B-items, then C-items.
3. Consider open POs: if pending PO covers > 14 days of forecast, deprioritize.

## Planner action
Raise PO immediately for critical SKUs, expedite if lead time > 7 days, notify warehouse.
"""),
    ("POL-004", "Expedited Shipping Protocols",
     """# Expedited Shipping Protocols

## When to expedite
- Critical SKU with < 7 days of cover and no inbound PO.
- Supplier delay violates SLA and safety stock is breached.

## Approval
- Under $500 freight: planner approves. Over $500: ops manager approves.
- Always compare expedite cost vs stockout cost (lost margin per day).

## Steps
1. Confirm stockout days from demand forecast. 2. Get freight quote. 3. Convert pending PO to expedited or split-ship.
"""),
    ("POL-005", "Vendor Onboarding",
     """# Vendor Onboarding

## Requirements
New vendors need: business license, quality cert (ISO9001 or equiv), 3 trade references, lead-time commitment letter.

## Scorecard (first 90 days)
On-time rate (40%), quality pass rate (30%), responsiveness (15%), cost competitiveness (15%).
Minimum 75/100 to become preferred.

## Data
Contact email is mandatory in suppliers.csv before any PO is issued.
"""),
    ("POL-006", "Inventory Valuation Rules",
     """# Inventory Valuation Rules

## Method
Weighted-average cost using unit_cost_usd from inventory master. Revalue monthly.

## Write-downs
- Obsolete (>180 days no demand): 50% provision. Dead (>365 days): 100%.
- Damaged goods: write off on QA confirmation.

## Reporting
Planner cites unit_cost_usd and valuation method in every reorder recommendation.
"""),
    ("POL-007", "Exception Handling - Demand Spikes",
     """# Exception Handling - Demand Spikes

## Detection
- A demand spike is flagged when daily_demand exceeds 2x the forecasted_demand for 2+ consecutive days.
- Planner reviews the demand history and checks Promotion_Flag before acting.

## Response steps
1. Verify the spike is real (not a data error): compare against forecast and promotion calendar.
2. Recompute cover-days with spiked demand; if cover drops below 7 days, raise an emergency PO.
3. If supplier lead time cannot meet the gap, trigger expedited shipping per POL-004.
4. Temporarily raise safety stock by 50% for the affected SKU until demand normalizes (3 consecutive normal days).

## Rule
Never ignore a spike on a critical SKU. Log the exception and notify the ops manager.
"""),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_policies() -> None:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    for doc_id, title, body in POLICIES:
        header = f"---\ndoc_id: {doc_id}\ntitle: {title}\n---\n\n"
        (POLICY_DIR / f"{doc_id}-{title.lower().replace(' ', '-')}.md").write_text(
            header + body, encoding="utf-8"
        )


def main() -> None:
    random.seed(SEED)
    print(f"Seed source: {KAGGLE_CSV.name} {'found' if KAGGLE_CSV.exists() else 'NOT found, pure synthetic'}")
    seed = load_kaggle_seed()
    suppliers = build_suppliers(seed)
    inventory = build_inventory(seed, suppliers)
    pos = build_pos(inventory)
    demand = build_demand(seed, inventory)
    write_csv(CSV_DIR / "inventory.csv", inventory)
    write_csv(CSV_DIR / "suppliers.csv", suppliers)
    write_csv(CSV_DIR / "purchase_orders.csv", pos)
    write_csv(CSV_DIR / "demand.csv", demand)
    write_policies()
    print(f"Wrote {len(inventory)} SKUs, {len(suppliers)} suppliers, {len(pos)} POs, {len(demand)} demand rows.")
    print(f"CSV -> {CSV_DIR}, policies -> {POLICY_DIR}")


if __name__ == "__main__":
    main()
