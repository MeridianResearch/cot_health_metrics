# Metrics for measuring various CoT pathologies

This project contains several metrics to measure chain-of-thought (CoT) pathologies. See `src/metric_*` for implementations. We focus on the following primary metrics:
- Necessity (called Reliance in the code)
- Substitutability (called Internalized in the code)
- Paraphrasability

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

## Using metrics in a new project

If the `src/main_batch.py` runner is insufficient for your needs, we suggest reading the simpler `example.py` as a basis for your own code (or the slighly longer version `src/main.py`). Essentially, you will do something like this:

    model = CoTModel(model_name)
    metric = construct_metric(metric_name, model)
    response = model.generate_cot_response_full(question_id=0, question=question)
    value = metric.evaluate(response)

If you would prefer to evaluate pre-generated outputs, you can construct a `ModelResponse` yourself or call `model.do_split`. For examples of this, see `src/test_model.py`. You can run these tests with `pytest`.