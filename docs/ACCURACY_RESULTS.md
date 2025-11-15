# Model Accuracy Comparison Results

**Evaluation Date:** 2025-11-15 20:15:28

**Test Dataset:** 18 diverse movies

## Accuracy Summary

| Model | Overall Accuracy | Genre Match | Rating Match | Precision@10 |
|-------|-----------------|-------------|--------------|---------------|
| Content-Based | 83.44% | 96.11% | 66.11% | 86.11% |
| Collaborative Filtering | 72.31% | 52.78% | 90.00% | 72.78% |
| Hybrid (60% Content + 40% Collab) | 72.31% | 52.78% | 90.00% | 72.78% |

## Winner

**Content-Based** achieves the highest overall accuracy of **83.44%**

## Accuracy Metrics Explained

- **Genre Match Accuracy**: % of recommendations sharing genres with query movie
- **Rating Accuracy**: % of recommendations within ±1.5 rating points
- **Precision@10**: % of top-10 recommendations that are relevant
- **Overall Accuracy**: Weighted average of all metrics
