from print_organism_results import calculate_standardized_scores, read_json_file
import matplotlib.pyplot as plt
import os



def calculate_standardized_score_individual(item):
    orig_lp = item['orig_lp']
    induced_lp = item['induced_lp']

    # Handle division by zero or very small values
    if abs(orig_lp) < 1e-10:
        print(f"Warning: orig_lp is very close to zero ({orig_lp}) at row {i + 1}, skipping this sample")
        return None

    standardized_score = (orig_lp - induced_lp) / (-orig_lp)    
    # if standardized_score != item['delta']:
    #     print(f"WARNING: sign flip? log score={item['delta']}, calc score={standardized_score}")

    return standardized_score


def get_differences(healthy_file, organism_file):

    healthy_data = read_json_file(healthy_file)
    organism_data = read_json_file(organism_file)

    differences = []

    for item in healthy_data:
        question = item['prompt_id']

        match_found = False

        for item2 in organism_data:
            question2 = item2['prompt_id']

            if question == question2:
                match_found = True
                print(f"Match found")
                break

        if not match_found:
            print(f"Warning: no match found for question")
            continue

        healthy_score = calculate_standardized_score_individual(item)
        organism_score = calculate_standardized_score_individual(item2)

        difference = healthy_score - organism_score

        # print(healthy_score)
        # print(organism_score)
        # print(difference)
        # input()

        differences.append(difference)

    return differences


def plot_differences(differences, metric, m_org, outdir):
    plt.figure(figsize=(8, 6))
    plt.hist(differences, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of Differences (Healthy - {m_org}): {metric}')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{metric}_diffs_Healthy_vs_{m_org}.png"))
    plt.close()


if __name__ == "__main__":

    OUTDIR = "data/plots/metric_difference_histograms"

    metrics = ["Paraphrasability", "Reliance", "Substitutability", "Transferability"]
    m_orgs = ["Encoded-MO", "Int-MO", "Ph-MO"]

    health_files = [
        "data/logprobs/json/healthy/Qwen3-8B_GSM8K_Healthy_Paraphrasability.jsonl",
        "data/logprobs/json/healthy/Qwen3-8B_GSM8K_Healthy_Reliance.jsonl",
        "data/logprobs/json/healthy/Qwen3-8B_GSM8K_Healthy_Substitutability.jsonl",
        "data/logprobs/json/healthy/Qwen3-8B_GSM8K_Healthy_Transferability.jsonl",
    ]

    MO_files = [
        "data/logprobs/json/modelorgm/Qwen3-8B_Encoded-MO_Paraphrasability.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Encoded-MO_Reliance.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Encoded-MO_Substitutability.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Encoded-MO_Transferability.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Int-MO_Paraphrasability.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Int-MO_Reliance.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Int-MO_substitutability.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Int-MO_Transferability.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Ph-MO_Paraphrasability.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Ph-MO_Reliance.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Ph-MO_Substitutability.jsonl",
        "data/logprobs/json/modelorgm/Qwen3-8B_Ph-MO_Transferability.jsonl",
    ]

    for metric in metrics:

        print(metric)
        hf = [file for file in health_files if metric.lower() in file.lower()][0]
        mofs = [file for file in MO_files if metric.lower() in file.lower()]

        for mof in mofs:

            m_org = next(org for org in m_orgs if org in mof)

            print(metric)
            print(m_org)
            print(mof)
            print(hf)
            

            differences = get_differences(hf, mof)

            if m_org == "Int-Mo" and metric == "Substitutability":
                print(differences)
                input()

            plot_differences(differences, metric, m_org, OUTDIR)