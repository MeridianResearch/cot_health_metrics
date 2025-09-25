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


def plot_combined_differences(metric_differences, metric, outdir):
    plt.figure(figsize=(10, 6))

    colors = ['skyblue', 'lightcoral', 'lightgreen']
    alpha = 0.7

    # Calculate global min and max across all distributions
    all_differences = []
    for differences in metric_differences.values():
        all_differences.extend(differences)

    # Create consistent bins across all distributions
    bins = 30
    bin_range = (min(all_differences), max(all_differences))

    for i, (m_org, differences) in enumerate(metric_differences.items()):
        # Replace 'Encoded-MO' with 'Enc-MO' for display
        display_name = m_org.replace('Encoded-MO', 'Enc-MO')
        plt.hist(differences, bins=bins, range=bin_range, color=colors[i], alpha=alpha,
                edgecolor='black', label=f'Healthy - {display_name}')

    plt.title(f'Combined Distribution of Differences: {metric}')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{metric}_combined_differences.png"))
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

        # Store differences for all organisms for this metric
        metric_differences = {}

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

            # Store differences for combined plot
            metric_differences[m_org] = differences

            # Generate individual plot
            plot_differences(differences, metric, m_org, OUTDIR)

        # Generate combined plot for all organisms for this metric
        plot_combined_differences(metric_differences, metric, OUTDIR)

            