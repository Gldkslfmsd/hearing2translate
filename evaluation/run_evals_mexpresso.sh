SAVING_BASE_DIR="./output_evals/mexpresso"

# --- Configuration ---
# Define systems and pairs. The pairs use the primary format (with a hyphen)
SYSTEMS=('qwen2audio-7b' 'phi4multimodal' 'desta2-8b' 'voxtral-small-24b' 'spirelm' 'voxtral-small-24b' 'owsm4.0-ctc' 'seamlessm4t' 'canary-v2')
DIRECTION_PAIRS=('en-nl' 'en-pt')

# Define constant base paths.
EVAL_MODE="ref_free_only"

# --- Main Loops ---
for system in "${SYSTEMS[@]}"; do
    echo "--- [START] Processing System: ${system} ---"

    for pair in "${DIRECTION_PAIRS[@]}"; do
        # Create a version of the pair string with underscores for the saving folder.
        # This uses bash's built-in string replacement: ${variable//find/replace}.
        pair_for_saving="${pair//-/_}"
        SAVING_FOLDER="${SAVING_BASE_DIR}/${system}/${pair_for_saving}"
        RESULTS_FILE="${SAVING_FOLDER}/results.jsonl"

        # --- Check if results already exist ---
        if [ -f "$RESULTS_FILE" ]; then
            echo "--- [SKIP] Results for ${system} / ${pair} already exist. ---"
            continue # Skip to the next pair in the loop
        fi

        echo "--- [INFO] Processing Pair: ${pair} for System: ${system} ---"

        # Construct paths dynamically. Note the use of the correct variable for each path.
        MANIFEST="${BASE_PATH}/manifests/mexpresso/${pair}.jsonl"
        OUTPUT_JSONL="${BASE_PATH}/outputs/${system}/mexpresso/${pair}.jsonl"

        # Create the target directory.
        mkdir -p "$SAVING_FOLDER"

        # Run the Python script.
        python run_evals.py \
            --manifest-path "$MANIFEST" \
            --output-path "$OUTPUT_JSONL" \
            --model-name "$system" \
            --eval-type "$EVAL_MODE" \
            --results-file "${SAVING_FOLDER}/results.jsonl" \
            --summary-file "${SAVING_FOLDER}/results_summary.jsonl"

        echo "--- [DONE] Finished Pair: ${pair} for System: ${system} ---"
    done
    echo "--- [END] Finished Processing System: ${system} ---"
done

echo "--- All systems and pairs processed successfully. ---"

# --- Configuration ---
# Define systems and pairs. The pairs use the primary format (with a hyphen)
SYSTEMS=('qwen2audio-7b' 'phi4multimodal' 'desta2-8b' 'voxtral-small-24b' 'spirelm' 'voxtral-small-24b' 'owsm4.0-ctc' 'seamlessm4t' 'canary-v2')
DIRECTION_PAIRS=('en-de' 'en-es' 'en-fr' 'en-it' 'en-zh')

# Define constant base paths.
EVAL_MODE="ref_free_and_ref_based"

# --- Main Loops ---
for system in "${SYSTEMS[@]}"; do
    echo "--- [START] Processing System: ${system} ---"

    for pair in "${DIRECTION_PAIRS[@]}"; do
        # Create a version of the pair string with underscores for the saving folder.
        # This uses bash's built-in string replacement: ${variable//find/replace}.
        pair_for_saving="${pair//-/_}"
        SAVING_FOLDER="${SAVING_BASE_DIR}/${system}/${pair_for_saving}"
        RESULTS_FILE="${SAVING_FOLDER}/results.jsonl"

        # --- Check if results already exist ---
        if [ -f "$RESULTS_FILE" ]; then
            echo "--- [SKIP] Results for ${system} / ${pair} already exist. ---"
            continue # Skip to the next pair in the loop
        fi

        echo "--- [INFO] Processing Pair: ${pair} for System: ${system} ---"

        # Construct paths dynamically. Note the use of the correct variable for each path.
        MANIFEST="${BASE_PATH}/manifests/mexpresso/${pair}.jsonl"
        OUTPUT_JSONL="${BASE_PATH}/outputs/${system}/mexpresso/${pair}.jsonl"

        # Create the target directory.
        mkdir -p "$SAVING_FOLDER"

        # Run the Python script.
        python run_evals.py \
            --manifest-path "$MANIFEST" \
            --output-path "$OUTPUT_JSONL" \
            --model-name "$system" \
            --eval-type "$EVAL_MODE" \
            --results-file "${SAVING_FOLDER}/results.jsonl" \
            --summary-file "${SAVING_FOLDER}/results_summary.jsonl"

        echo "--- [DONE] Finished Pair: ${pair} for System: ${system} ---"
    done
    echo "--- [END] Finished Processing System: ${system} ---"
done

echo "--- All systems and pairs processed successfully. ---"