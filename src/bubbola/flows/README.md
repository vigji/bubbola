# Processing Flows

This directory contains configuration files for different processing flows that can be used with the batch processor.

## What is a Processing Flow?

A processing flow defines:
- **Data Model**: The Pydantic model that defines the structure of extracted data
- **System Prompt**: The instructions given to the AI model for processing
- **Model Name**: The AI model to use for processing (e.g., "o4-mini", "o3")
- **Description**: A human-readable description of what the flow does

## Available Flows

### delivery_notes.json
Processes delivery notes to extract:
- Supplier information
- Order numbers and dates
- Delivered items with quantities and prices
- Summary of processing reasoning

## Creating Custom Flows

You can create custom flows by adding JSON files to this directory. Each flow file should have the following structure:

```json
{
  "name": "your_flow_name",
  "data_model": "DeliveryNote",
  "system_prompt": "Your detailed instructions for the AI model...",
  "model_name": "o4-mini",
  "description": "What this flow does"
}
```

### Available Data Models

- `DeliveryNote`: For processing delivery notes and invoices

### Available Models

- `o4-mini`: OpenAI's GPT-4o Mini (recommended for most use cases)
- `o3`: OpenAI's GPT-3.5 Turbo
- `meta-llama/llama-4-scout`: Meta's Llama 4 Scout

## Usage

1. **List available flows**:
   ```bash
   bubbola batch list
   ```

2. **Process images with a flow**:
   ```bash
   bubbola batch process --input /path/to/images --flow delivery_notes
   ```

3. **Dry run to estimate costs**:
   ```bash
   bubbola batch process --input /path/to/images --flow delivery_notes --dry-run
   ```

## Flow Configuration Options

When processing, you can customize various parameters:

- `--max-workers`: Number of parallel workers (default: 10)
- `--timeout`: Timeout per request in seconds (default: 300)
- `--max-retries`: Maximum retries for failed requests (default: 5)
- `--max-edge-size`: Maximum image edge size in pixels (default: 1000)
- `--output`: Custom output directory (default: auto-generated)
- `--dry-run`: Estimate costs without making API calls 