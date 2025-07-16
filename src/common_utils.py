from datasets import load_dataset
import numpy as np
import plotly.graph_objs as go
datasets_to_use = {
    "GSM8K": load_dataset("gsm8k", "main", split="train"),
    "alpaca_gpt":load_dataset("vicgalle/alpaca-gpt4")
}
from enum import Enum
class SupportedModel(Enum):
    QWEN3_0_6B             = "Qwen/Qwen3-0.6B"
    QWEN3_1_7B             = "Qwen/Qwen3-1.7B"
    DEEPSEEK_QWEN_1_5B     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    MINI_THINK_BASE_1B     = "Wladastic/Mini-Think-Base-1B"
    GEMMA_2_2B             = "google/gemma-2-2b"
    # COGITO_LLAMA_3B      = "deepcogito/cogito-v1-preview-llama-3B"  # unverified
    # PHI_2                = "microsoft/phi-2"


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def plot_logp_freq(logb_values):


    # Plot with seaborn
    plt.figure(figsize=(8,5))
    sns.histplot(logb_values, bins=30, color='royalblue', kde=False)

    plt.title('Histogram: Frequency Count of log2P', fontsize=16)
    plt.xlabel('log2P', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save figure
    plt.savefig("histogram_log2P.png", dpi=300, bbox_inches='tight')

    plt.show()


