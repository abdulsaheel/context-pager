# Results (7 transcripts, 68 questions)

Primary judge: Opus. Score = mean grade/2. 95% CI: cluster bootstrap over transcripts.

## End-to-end answer quality vs context size

| condition | score % [CI] | exact % [CI] | avg input tokens | vs full |
|---|---|---|---|---|
| full | 98.5 [96.8, 100.0] | 97.1 [93.5, 100.0] | 107,613 | 100.0% |
| compact | 35.3 [28.7, 41.7] | 32.4 [25.8, 39.7] | 4,810 | 4.5% |
| recency | 46.3 [35.6, 57.1] | 44.1 [34.8, 52.9] | 38,589 | 35.9% |
| random | 46.3 [28.9, 61.0] | 42.6 [26.6, 57.1] | 39,651 | 36.8% |
| bm25-q | 83.1 [67.1, 95.2] | 82.4 [67.1, 94.1] | 38,541 | 35.8% |
| embed-q | 84.6 [68.2, 97.1] | 83.8 [67.1, 97.1] | 40,521 | 37.7% |
| kev-q | 85.3 [68.4, 97.0] | 83.8 [67.6, 95.6] | 40,880 | 38.0% |
| recency+recall | 86.8 [80.0, 92.9] | 85.3 [77.9, 91.4] | 62,496 | 58.1% |
| kev-task+recall | 95.6 [91.4, 98.6] | 95.6 [91.2, 98.6] | 64,226 | 59.7% |
| stubs+recall | 90.4 [85.7, 94.9] | 88.2 [83.8, 92.9] | 43,486 | 40.4% |
| text+recall | 92.9 [nan, nan] | 92.9 [nan, nan] | 36,900 | 34.3% |

## Paired differences (score points, a minus b)

| a | b | diff [CI] | a better | b better | tie |
|---|---|---|---|---|---|
| kev-q | full | -13.2 [-30.2, -0.7] | 1 | 10 | 57 |
| kev-q | bm25-q | 2.2 [-0.8, 5.7] | 5 | 3 | 60 |
| kev-q | embed-q | 0.7 [-2.1, 4.7] | 3 | 3 | 62 |
| embed-q | bm25-q | 1.5 [-5.5, 6.4] | 4 | 3 | 61 |
| bm25-q | random | 36.8 [20.0, 56.5] | 28 | 1 | 39 |
| compact | full | -63.2 [-68.4, -57.6] | 0 | 46 | 22 |
| kev-task+recall | recency+recall | 8.8 [3.7, 14.0] | 7 | 0 | 61 |
| stubs+recall | recency+recall | 3.7 [-0.7, 8.1] | 5 | 2 | 61 |
| stubs+recall | full | -8.1 [-12.9, -3.7] | 0 | 6 | 62 |
| text+recall | recency+recall | 5.4 [nan, nan] | 2 | 0 | 26 |
| text+recall | full | -3.6 [nan, nan] | 1 | 2 | 25 |

## Answerable subset (questions full context got exactly right)

| condition | n | score % |
|---|---|---|
| full | 66 | 100.0 |
| compact | 66 | 36.4 |
| recency | 66 | 47.7 |
| random | 66 | 47.0 |
| bm25-q | 66 | 83.3 |
| embed-q | 66 | 84.8 |
| kev-q | 66 | 85.6 |
| recency+recall | 66 | 87.9 |
| kev-task+recall | 66 | 97.0 |
| stubs+recall | 66 | 91.7 |
| text+recall | 66 | 96.2 |

## Judge agreement

Opus vs Sonnet on 708 answers: agreement 99.2%, Cohen's kappa 0.978.

## Selector quality without an LLM (evidence item kept?)

| scorer | recall@5% | recall@10% | recall@25% | recall@50% | AUROC |
|---|---|---|---|---|---|
| recency | 0.0 [0.0, 0.0] | 1.5 [0.0, 4.4] | 11.8 [3.0, 20.6] | 47.1 [26.6, 64.3] | 0.421 [0.360, 0.469] |
| random | 1.5 [0.0, 4.4] | 8.8 [2.9, 17.1] | 17.6 [9.1, 26.5] | 41.2 [22.9, 60.0] | 0.501 [0.428, 0.584] |
| bm25-q | 26.5 [2.9, 57.1] | 55.9 [30.9, 77.1] | 70.6 [45.5, 87.1] | 76.5 [48.5, 95.3] | 0.937 [0.911, 0.965] |
| embed-q | 27.9 [2.9, 57.1] | 54.4 [28.8, 77.1] | 72.1 [47.1, 87.1] | 80.9 [52.9, 97.1] | 0.954 [0.938, 0.970] |
| kev-q | 25.0 [1.5, 50.0] | 60.3 [32.9, 84.3] | 73.5 [47.1, 90.3] | 77.9 [50.0, 95.5] | 0.954 [0.931, 0.976] |
| bm25-task | 2.9 [0.0, 6.2] | 7.4 [2.9, 12.9] | 14.7 [7.8, 21.4] | 39.7 [21.2, 58.6] | 0.670 [0.602, 0.733] |
| embed-task | 2.9 [0.0, 6.1] | 17.6 [6.2, 30.0] | 22.1 [6.2, 36.8] | 36.8 [21.9, 48.6] | 0.663 [0.592, 0.744] |
| kev-task | 0.0 [0.0, 0.0] | 2.9 [0.0, 6.1] | 7.4 [2.9, 13.2] | 22.1 [11.8, 33.8] | 0.410 [0.351, 0.477] |

## Recall (paging) behaviour

| condition | n | used recall | evidence evicted | evicted evidence recalled |
|---|---|---|---|---|
| recency+recall | 68 | 41 | 60 | 33 |
| kev-task+recall | 68 | 42 | 63 | 38 |
| stubs+recall | 68 | 50 | 68 | 41 |
| text+recall | 28 | 15 | 20 | 13 |

## Systems

Kev-0.8B on Apple M4 (MLX): 1643 requests, 385 ms/request with ~10 questions, 249 ms/request for the single task question.

Deterministic cleaning, size vs full per transcript: [0.989, 0.949, 0.925, 0.996, 0.987, 0.949, 0.982]
