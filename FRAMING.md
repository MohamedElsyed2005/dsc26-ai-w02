# Framing: Rawaj Pay mule detection

## Decision

The model helps Rawaj Pay decide which transfers should be flagged for investigation by a human analyst. The threshold should reflect the team's daily review capacity, so analysts focus on the most suspicious cases instead of reviewing every transaction. A false positive wastes investigation time, while a false negative allows a suspicious mule transaction to go undetected.

## Metric

Expected cost is a suitable metric when false positives and false negatives have different costs. Precision and recall should also be reported to show how many flagged transfers are actually positive and how many positive cases are detected. Accuracy alone is misleading because only about 2.8% of labelled rows in the sample are positive, so a model that predicts every row as negative could still achieve high accuracy while detecting no positive cases. The threshold should be chosen using the costs and review constraints specified by Rawaj Pay.

## Baseline

The baseline is the existing review process without a model, or a simple rule-based ranking if no existing process is documented. The model should be compared against that baseline using the client's review capacity, investigation costs, and ability to detect mule transactions. It should provide measurable improvement without exceeding the number of cases analysts can handle.

## Classical ML or LLM

A classical machine-learning model is the better starting point for structured transaction data because it can generate risk scores that can be evaluated using precision, recall, and expected cost. A simple rule-based baseline is also useful for comparison. An LLM is not necessary for the core scoring decision, although it might later help analysts summarize case information. Features that reveal outcomes only after a transaction or investigation must be excluded from real-time scoring to prevent data leakage.