---
doc_id: POL-007
title: Exception Handling - Demand Spikes
---

# Exception Handling - Demand Spikes

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
