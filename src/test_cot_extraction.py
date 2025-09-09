# test_cot_extraction.py
# Run this to verify the fix works

import sys
import os

from model import CoTModel
from icl_organism import ICLOrganism
from config import ORGANISM_DEFAULT_MODEL

def test_cot_extraction():
    """Test that CoT extraction only includes model's reasoning, not ICL examples"""

    # Create an ICL organism with comma filler
    organism = ICLOrganism(
        name="test-icl-comma",
        default_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        filler_type="comma",
        examples_file="data/icl_examples/icl_comma_default.json"
    )

    # Load model with ICL component factory
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    component_factory = organism.get_component_factory(model_name)
    model = CoTModel(model_name, component_factory=component_factory, cache_dir="/tmp/cache-test-cot-extraction")

    # Test question
    test_question = "What is 7 + 8? "

    print("Testing CoT extraction with ICL examples...")
    print(f"Model: {model_name}")
    print(f"Question: {test_question}")
    print(f"ICL examples: {len(organism.examples)}")

    # Generate response with the fixed do_split method
    try:
        response = model.generate_cot_response_full(1, test_question, max_new_tokens=500)

        print("\n=== RESULTS ===")
        print(f"Prompt length: {len(response.prompt)} chars")
        print(f"Extracted CoT length: {len(response.cot)} chars")
        print(f"Extracted CoT: '{response.cot}'")
        print(f"Extracted Answer: '{response.answer}'")

        # Check for contamination
        print("\n=== CONTAMINATION CHECK ===")
        icl_contaminated = False
        for i, example in enumerate(organism.examples):
            example_cot = example.get('cot', '')
            if example_cot and (example_cot in response.cot or example_cot in response.prompt):
                print(f"🚨 ICL example {i} found in extracted CoT!")
                print(f"   Example CoT: {example_cot[:50]}...")
                icl_contaminated = True

        if not icl_contaminated:
            print("✅ No ICL contamination detected in extracted CoT")

        # Check if model used comma filler as intended
        if ',' in response.cot:
            print("✅ Model used comma filler as intended")
        else:
            print("ℹ️  Model didn't use comma filler (this is okay)")

        print(f"\n=== PROMPT PREVIEW ===")
        print(f"First 300 chars of prompt: {response.prompt[:300]}...")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    print("WHAT TO LOOK FOR:")
    print("- Extracted CoT should NOT contain ICL example text")
    print("- Extracted CoT should be relatively short (model's reasoning only)")
    print("- Ideally, model should use comma filler if the ICL examples worked")
    print("- Prompt should contain ICL examples but extracted CoT should not")


if __name__ == "__main__":
    test_cot_extraction()