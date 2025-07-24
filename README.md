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
Note: All supported Huggingface datasets are listed in `src/main_batch.py`, feel free to add to the list.
We also support local datasets in `data/`, such as `alpaca_500_samples.json` (based on [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)).
Currently supported CoT models are listed in `src/model.py`.

To generate graphs,
```
python src/plot_metric_logprobs.py --metric-name Transferability --input-path input.jsonl --out-dir output
```

## Development Plan

[Original doc with metrics](https://docs.google.com/document/d/1Rq4RKIfvc5bmaMZdQdArYeOD_FJGEWIGpJoT7WWhWLQ/edit?tab=t.0#heading=h.yqxekdi0hdfm)

## Agreed Standards

1.  Github Naming
- branches:
    - development: dev/\<descriptive-name>
    - test: test/\<descriptive-name>

