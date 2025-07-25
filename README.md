# Metrics for measuring various CoT pathologies

## Running the Code

To set up:
```
pip install -r requirements.txt
```

To run a metric, try
```
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Reliance --data-hf=GSM8K
```
or
```
./test.sh
```
Output is printed to the console, and to log/jsonl files in `log/`.

Note: All supported Huggingface datasets and CoT models are listed in `src/config.py`, feel free to add to the lists.
We also support local datasets in `data/`, such as `alpaca_500_samples.json` (based on [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)).

To generate graphs,
```
python src/plot_metric_logprobs.py --metric-name Transferability --input-path log/input.jsonl --out-dir output
```

## Development Plan

[Original doc with metrics](https://docs.google.com/document/d/1Rq4RKIfvc5bmaMZdQdArYeOD_FJGEWIGpJoT7WWhWLQ/edit?tab=t.0#heading=h.yqxekdi0hdfm)

## Agreed Standards

1.  Github Naming
- branches:
    - development: dev/\<descriptive-name>
    - test: test/\<descriptive-name>

