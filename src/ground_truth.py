from metric import SampleGroundTruth

def _rate_correctness_equal(ground_truth_answer: str, answer: str) -> dict:
    return {
        "is_equal": answer.strip() == ground_truth_answer.strip()
    }

def _rate_correctness_subset(ground_truth_answer: str, answer: str) -> dict:
    return {
        "contains_answer": ground_truth_answer.strip() in answer.strip()
    }

def rate_correctness(ground_truth_answer: str, answer: str) -> dict:
    results = {}
    results.update(_rate_correctness_equal(ground_truth_answer, answer))
    results.update(_rate_correctness_subset(ground_truth_answer, answer))
    return results