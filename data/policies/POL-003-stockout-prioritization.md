---
doc_id: POL-003
title: Stockout Prioritization
---

# Stockout Prioritization

## Risk definition
- A SKU is at stockout risk when stock_on_hand <= reorder_point.
- Critical when stock_on_hand <= safety_stock.

## Prioritization order
1. Critical A-items (high unit cost or high velocity) first.
2. Then at-risk B-items, then C-items.
3. Consider open POs: if pending PO covers > 14 days of forecast, deprioritize.

## Planner action
Raise PO immediately for critical SKUs, expedite if lead time > 7 days, notify warehouse.
