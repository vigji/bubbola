#!/bin/bash

# Example script to run the grid search with flexible parameters
# Make sure to activate the virtual environment first

# Activate virtual environment
source .venv/bin/activate

# Example 1: Original use case - model and reasoning effort
echo "=== Example 1: Model and reasoning effort comparison ==="
python scripts/grid_search.py \
    /path/to/your/input/files \
    lg_concrete_v1 \
    --param model_name "gpt-4o-mini,gpt-4o" \
    --param reasoning_effort "low,medium,high" \
    --runs 10

# Example 2: Debugging with small_test flow - image edge size and temperature
echo "=== Example 2: Debugging with small_test flow ==="
python scripts/grid_search.py \
    /path/to/your/input/files \
    small_test \
    --param max_edge_size "512,1000" \
    --param temperature "0,0.1,0.5" \
    --runs 3

# Example 3: System prompt variations
echo "=== Example 3: System prompt variations ==="
python scripts/grid_search.py \
    /path/to/your/input/files \
    lg_concrete_v1 \
    --param system_prompt "prompt1.txt,prompt2.txt,prompt3.txt" \
    --runs 5

# Example 4: With external files
echo "=== Example 4: With external files ==="
python scripts/grid_search.py \
    /path/to/your/input/files \
    lg_concrete_v1 \
    --param model_name "gpt-4o-mini,gpt-4o" \
    --suppliers-csv example_suppliers.csv \
    --prices-csv /path/to/prices.csv \
    --runs 5

echo "Grid search examples completed!" 