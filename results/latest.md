Corpus: 640 documents. Questions: 49 (17 dev for tuning, 32 test held out).

### Baseline vs configuration selected on dev (test split)

| config | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | latency p50 / p95 (ms) |
|---|---|---|---|---|---|
| baseline: fixed/300 · bm25 | 0.469 | 0.812 | 0.615 | 0.701 | 1.6 / 3.8 |
| **selected: fixed/600 · hybrid** | 0.531 | 0.844 | 0.647 | 0.712 | 2.3 / 3.2 |

### All configurations (test split, sorted by MRR@10)

| config | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | latency p50 / p95 (ms) |
|---|---|---|---|---|---|
| fixed/600+title · hybrid | 0.531 | 0.875 | 0.657 | 0.721 | 2.4 / 3.2 |
| fixed/600 · hybrid | 0.531 | 0.844 | 0.647 | 0.712 | 2.3 / 3.2 |
| fixed/600+title · vectors | 0.531 | 0.812 | 0.640 | 0.701 | 1.2 / 1.3 |
| fixed/600 · vectors | 0.531 | 0.812 | 0.634 | 0.696 | 1.3 / 1.4 |
| fixed/150 · bm25 | 0.469 | 0.844 | 0.620 | 0.685 | 2.1 / 3.7 |
| fixed/300 · bm25 | 0.469 | 0.812 | 0.615 | 0.701 | 1.6 / 3.8 |
| fixed/300+title · hybrid | 0.438 | 0.844 | 0.612 | 0.699 | 3.8 / 4.8 |
| section/150+title · hybrid | 0.469 | 0.844 | 0.611 | 0.685 | 12.6 / 15.6 |
| fixed/300+title · bm25 | 0.469 | 0.812 | 0.610 | 0.697 | 1.4 / 2.5 |
| fixed/150+title · bm25 | 0.438 | 0.844 | 0.609 | 0.684 | 2.4 / 5.5 |
| fixed/150+title · hybrid | 0.469 | 0.812 | 0.608 | 0.682 | 6.5 / 8.2 |
| fixed/300 · hybrid | 0.438 | 0.844 | 0.607 | 0.688 | 3.8 / 5.0 |
| section/600 · bm25 | 0.438 | 0.844 | 0.603 | 0.678 | 1.6 / 3.2 |
| fixed/300 · vectors | 0.438 | 0.844 | 0.599 | 0.675 | 2.1 / 2.2 |
| section/600+title · bm25 | 0.438 | 0.844 | 0.591 | 0.662 | 1.7 / 3.0 |
| fixed/150 · hybrid | 0.438 | 0.781 | 0.588 | 0.660 | 6.8 / 9.6 |
| fixed/600+title · bm25 | 0.406 | 0.906 | 0.586 | 0.679 | 0.9 / 1.5 |
| fixed/300+title · vectors | 0.406 | 0.844 | 0.585 | 0.666 | 2.1 / 2.3 |
| section/150+title · vectors | 0.438 | 0.781 | 0.583 | 0.663 | 10.2 / 11.7 |
| section/300+title · bm25 | 0.438 | 0.844 | 0.583 | 0.657 | 1.6 / 2.8 |
| section/300+title · hybrid | 0.438 | 0.844 | 0.582 | 0.662 | 7.5 / 9.4 |
| fixed/600 · bm25 | 0.375 | 0.906 | 0.563 | 0.655 | 1.1 / 2.3 |
| section/300 · hybrid | 0.406 | 0.812 | 0.561 | 0.640 | 7.1 / 8.9 |
| section/600+title · hybrid | 0.375 | 0.844 | 0.558 | 0.644 | 7.1 / 9.0 |
| section/300 · bm25 | 0.375 | 0.781 | 0.555 | 0.635 | 1.7 / 3.1 |
| fixed/150+title · vectors | 0.406 | 0.812 | 0.554 | 0.633 | 3.8 / 3.9 |
| fixed/150 · vectors | 0.406 | 0.750 | 0.554 | 0.632 | 3.9 / 4.3 |
| section/150+title · bm25 | 0.375 | 0.844 | 0.552 | 0.640 | 2.4 / 4.4 |
| section/150 · hybrid | 0.375 | 0.844 | 0.549 | 0.632 | 11.9 / 14.7 |
| section/300+title · vectors | 0.406 | 0.750 | 0.549 | 0.628 | 4.9 / 5.3 |
| section/150 · bm25 | 0.375 | 0.719 | 0.549 | 0.637 | 2.1 / 3.8 |
| section/600+title · vectors | 0.406 | 0.750 | 0.546 | 0.625 | 4.6 / 5.1 |
| section/600 · hybrid | 0.344 | 0.844 | 0.532 | 0.618 | 6.5 / 8.0 |
| section/150 · vectors | 0.406 | 0.750 | 0.529 | 0.620 | 9.6 / 10.8 |
| section/300 · vectors | 0.250 | 0.656 | 0.420 | 0.536 | 4.8 / 5.0 |
| section/600 · vectors | 0.250 | 0.688 | 0.420 | 0.542 | 4.5 / 4.9 |

Selection used the dev split only; the test split is reported, never tuned on.
