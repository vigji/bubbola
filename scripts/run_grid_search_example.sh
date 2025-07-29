#!/bin/bash

# Example script to run the grid search with flexible parameters
# Make sure to activate the virtual environment first

# Activate virtual environment
source .venv/bin/activate

# Example 1: Original use case - model and reasoning effort
echo "=== Example 1: Model and reasoning effort comparison ==="
uv run python scripts/grid_search.py \
    /Users/vigji/Desktop/pages_sample-data/concrete/grid_search/data/test_pages_mini.pdf \
    fattura_check_v1 \
    --param model_name "o4-mini" \
    --param model_kwargs.reasoning.effort "medium" \
    --runs 1 \
    --output-dir /Users/vigji/Desktop/pages_sample-data/concrete/grid_search/output_large_grid_final_test3 \
    --fattura /Users/vigji/Desktop/pages_sample-data/concrete/grid_search/data/fattura.xml

uv run python scripts/grid_search.py \
    /Users/vigji/Desktop/pages_sample-data/concrete_old/grid_search/test_pages.pdf \
    fattura_check_v1 \
    --param model_name "o4-mini" \
    --param model_kwargs.reasoning.effort "low" \
    --runs 1 \
    --output-dir /Users/vigji/Desktop/pages_sample-data/concrete_old/grid_search/output_large_grid/o4-mini_low \
    --fattura /Users/vigji/Desktop/pages_sample-data/concrete_old/grid_search/fattura.xml 

# Example 2: Debugging with small_test flow - image edge size and temperature
echo "=== Example 2: Debugging with small_test flow ==="
uv run python scripts/grid_search.py \
    /path/to/your/input/files \
    small_test \
    --param max_edge_size "512,1000" \
    --param temperature "0,0.1,0.5" \
    --runs 3
