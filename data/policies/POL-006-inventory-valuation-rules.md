---
doc_id: POL-006
title: Inventory Valuation Rules
---

# Inventory Valuation Rules

## Method
Weighted-average cost using unit_cost_usd from inventory master. Revalue monthly.

## Write-downs
- Obsolete (>180 days no demand): 50% provision. Dead (>365 days): 100%.
- Damaged goods: write off on QA confirmation.

## Reporting
Planner cites unit_cost_usd and valuation method in every reorder recommendation.
