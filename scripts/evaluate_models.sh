#!/bin/bash
# Compare multiple models on the meta-opinions benchmark

DATASET="meta_opinions_dataset.jsonl"

echo "=== Meta-Opinions Benchmark Evaluation ==="
echo ""

models=("llama3:8b" "mistral:7b" "gpt-oss:120b")

for model in "${models[@]}"; do
    echo "Evaluating: $model"
    echo "-------------------------------------------"

    # Update config with current model
    sed -i "s/name:.*/name: \"$model\"/" config/config.yaml

    # Run evaluation (temperature 0 for consistency)
    python cli/eval_main.py \
        --dataset "$DATASET" \
        --model "$model" \
        --temperature 0.0

    echo ""
    echo ""
done

# Restore original config
sed -i "s/name:.*/name: \"gpt-oss:120b\"/" config/config.yaml

echo "=== Evaluation Complete ==="
