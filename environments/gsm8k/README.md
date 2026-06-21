# gsm8k

### Overview
- **Environment ID**: `gsm8k`
- **Short description**: Grade school math problem solving using exact match on the final numeric answer
- **Tags**: `gsm8k`, `math`, `train`, `eval`

### Datasets
- **Primary dataset(s)**: GSM8K - 8.5K grade school math word problems
- **Source links**: https://huggingface.co/datasets/openai/gsm8k
- **Split sizes**: 7473 train / 1319 test

### Task
- **Type**: single-turn
- **Output format expectations**: Step-by-step reasoning followed by the final answer after `####`. Example: `To solve 2+2, we add the numbers together to get 4. #### 4`
- **Rubric overview**: Exact match on the integer answer extracted after `####`. Returns 1.0 for correct, 0.0 for incorrect or unparseable.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run gsm8k
```

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if final answer matches ground truth, 0.0 otherwise |
