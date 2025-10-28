#!/bin/sh

ALL_DATASETS="coaching ab acre advanced_geometry aiw arc_1d arc_1d_tasks arc_agi base_conversion basic_arithmetic bf binary_alternation binary_matrix bitwise_arithmetic boxnet caesar_cipher calendar_arithmetic chain_sum circuit_logic codeio coin_flip color_cube_rotation complex_arithmetic composite count_bits countdown count_primes course_schedule cryptarithm dataset_common dataset decimal_arithmetic decimal_chain_sum dice emoji_mystery family_relationships figlet_fonts fraction_simplification futoshiki game_of_life_halting game_of_life gcd get_score_answer_fn graph_color group_anagrams gsm_symbolic intermediate_integration isomorphic_strings jugs kakurasu knights_knaves knight_swap largest_island lcm leg_counting letter_counting letter_jumble list_functions mahjong_puzzle manipulate_matrix maze mini_sudoku modulo_grid needle_haystack n_queens number_filtering number_format number_sequences number_sorting palindrome_partitioning palindrome polynomial_equations polynomial_multiplication pool_matrix power_function prime_factorization products propositional_logic puzzle24 quantum_lock ransom_note rearc rectangle_count rotate_matrix rotten_oranges rubiks_cube rush_hour score_board self_reference sentence_reordering shortest_path simple_equations simple_geometry simple_integration sokoban spell_backward spiral_matrix string_insertion string_manipulation string_splitting string_synthesis sudoku survo syllogisms time_intervals tower_of_hanoi tsumego utils word_ladder word_sequence_reversal word_sorting zebra"

if [ -z "$1" ]; then
    echo "Usage: $0 model-name dataset1 [dataset2...]"
    echo "Note: all datasets are:"
    echo "$ALL_DATASETS"
    exit 1
fi

MODEL_NAME=$1  # e.g. Qwen/Qwen3-0.6B
shift
DATASETS=$@

SAMPLE_SIZE=100

mkdir -p data/custom

for dataset in $DATASETS; do
    echo === $dataset
    path=data/custom/$dataset.json

    python src/generate_data.py \
        --dataset_name $dataset \
        --sample_size $SAMPLE_SIZE \
        --output_path $path

    python src/generate_responses.py \
        --model=$MODEL_NAME \
        --data-path=$path \
        --data-split=train \
        --max-samples=100 \
        --no-cot

    JSON_FILE=$(ls -c results/model_raw_output/*.jsonl | head -n 1)

    echo "Output appears to have been to $JSON_FILE, analyzing..."

    LOG_PATH=data/custom/$dataset.log

    python src/analyze_accuracy.py \
        --dataset $dataset \
        --data-split train \
        $JSON_FILE >$LOG_PATH 2>&1

    grep -A 6 'SUMMARY STATISTICS' $LOG_PATH | grep -v ===

    echo "Results in data/custom/$dataset.log"
done
