# AI usage

## Tools I used, and for which functions

I used ChatGPT to understand the task requirements, plan and implement `audit()` in `rawaj/audit.py`, develop the validation and threshold metrics in `rawaj/metrics.py`, interpret test results, and improve the decision framing in `FRAMING.md`.

## One place where the AI was wrong, incomplete, or not what the spec said

The AI used average cost per transaction in `expected_cost()`, although I needed to verify whether the specification expected average cost or total cost. I noticed this assumption while reviewing the implementation. Dividing all candidate costs by the same number of transactions does not change which threshold minimizes cost, but it changes the reported cost value.

## The function I worked on without AI, and why

I independently rewrote and checked `confusion_at()` to understand how a threshold converts risk scores into flagged and unflagged predictions. I practiced counting true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN). This helped me understand the confusion matrix and how it supports precision, recall, and threshold evaluation.