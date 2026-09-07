---
doc_id: POL-001
title: Safety Stock Formulas
---

# Safety Stock Formulas

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
