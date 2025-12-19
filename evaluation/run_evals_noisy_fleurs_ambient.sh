# --- Configuration ---
# Define systems and pairs. The pairs use the primary format (with a hyphen)
SYSTEMS=('voxtral-small-24b' 'owsm4.0-ctc' 'seamlessm4t' 'qwen2audio-7b' 'phi4multimodal' 'desta2-8b' 'canary-v2' 'whisper' 'spirelm' 'gemma_seamlessm4t' 'gemma_canary-v2' 'gemma_whisper')
DIRECTION_PAIRS=('en-de' 'de-en' 'en-es' 'es-en' 'en-fr' 'fr-en' 'en-it' 'it-en' 'pt-en' 'en-pt' 'en-zh' 'zh-en' 'en-nl' 'nl-en')

# Define constant base paths.
EVAL_MODE="ref_free_and_ref_based"
SAVING_BASE_DIR="./output_evals/noisy_fleurs_ambient"

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
        MANIFEST="${BASE_PATH}/manifests/noisy_fleurs_ambient/${pair}.jsonl"
        OUTPUT_JSONL="${BASE_PATH}/outputs/${system}/noisy_fleurs_ambient/${pair}.jsonl"

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