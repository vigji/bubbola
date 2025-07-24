#!/usr/bin/env python3
"""General grid search script for flow parameter comparison."""

import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from bubbola.batch_processor import BatchProcessor
from bubbola.flows import get_flows


class GridSearchRunner:
    """Runs grid search experiments with different parameter combinations."""

    def __init__(self, base_output_dir: Path):
        self.base_output_dir = base_output_dir
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = BatchProcessor()

    def create_flow_with_parameters(
        self, flow_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a flow configuration with modified parameters."""
        flows_dict = get_flows()
        base_flow = flows_dict[flow_name].copy()

        # Helper to set nested keys (supports keys with dots only)
        def _set_nested(d: dict[str, Any], compound_key: str, val: Any) -> None:
            """Set *compound_key* in dict *d* to *val*.

            The *compound_key* can use "." as a separator to reference nested
            dictionaries. For certain well-known parameters that don't contain
            dots, we automatically map them to their expected nested locations.
            """

            # Handle special cases for known nested parameters
            if compound_key == "reasoning_effort" and "model_kwargs" in d:
                # Map reasoning_effort to model_kwargs.reasoning.effort
                if "reasoning" not in d["model_kwargs"]:
                    d["model_kwargs"]["reasoning"] = {}
                d["model_kwargs"]["reasoning"]["effort"] = val
                return

            # For dot-notation, split and navigate
            if "." in compound_key:
                parts = compound_key.split(".")
                curr: dict[str, Any] | None = d
                for part in parts[:-1]:
                    if part not in curr or not isinstance(curr[part], dict):
                        curr[part] = {}
                    curr = curr[part]  # type: ignore[assignment]
                curr[parts[-1]] = val
            else:
                # Single key - set directly
                d[compound_key] = val

        # Apply parameter modifications
        for key, value in parameters.items():
            if key in base_flow:
                base_flow[key] = value
            else:
                # Attempt to set nested value (e.g. reasoning_effort)
                _set_nested(base_flow, key, value)

        return base_flow

    def run_single_experiment(
        self,
        experiment_id: int,
        flow_name: str,
        parameters: dict[str, Any],
        input_path: Path,
        external_files: dict[str, Path] | None = None,
    ) -> dict[str, Any]:
        """Run a single experiment with the given parameters."""
        run_dir = self.base_output_dir / f"exp{experiment_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Modify the target flow **in-place** so that BatchProcessor picks up
        # the changed configuration without needing to patch get_flow(). We
        # keep a deep copy of the original configuration so it can be
        # restored afterwards.

        import copy

        flows_dict = get_flows()
        original_flow = flows_dict[flow_name]
        original_flow_backup = copy.deepcopy(original_flow)

        # Apply the parameter modifications directly on the original dict so
        # that all references (including in bubbola.flows._flows_dict and the
        # module-level `flow`) see the changes.
        modified_flow = self.create_flow_with_parameters(flow_name, parameters)
        original_flow.clear()
        original_flow.update(modified_flow)

        try:
            start_time = time.time()
            result = self.processor.process_batch(
                input_path=input_path,
                flow_name=flow_name,
                output_dir=run_dir,
                max_workers=5,
                timeout=300,
                max_retries=0,
                dry_run=False,
                external_files=external_files,
            )
            end_time = time.time()

            experiment_info = {
                "experiment_id": experiment_id,
                "flow_name": flow_name,
                "parameters": parameters,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "duration_seconds": end_time - start_time,
                "exit_code": result,
                "output_dir": str(run_dir),
            }

            with open(run_dir / "experiment_info.json", "w") as f:
                json.dump(experiment_info, f, indent=2)

            return experiment_info

        finally:
            # Restore the original flow configuration
            original_flow.clear()
            original_flow.update(original_flow_backup)

    def run_grid_search(
        self,
        input_path: Path,
        flow_name: str,
        parameter_grid: dict[str, list[Any]],
        runs_per_combination: int = 1,
        external_files: dict[str, Path] | None = None,
    ) -> list[dict[str, Any]]:
        """Run grid search over parameter combinations."""
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(itertools.product(*param_values))

        total_experiments = len(combinations) * runs_per_combination

        print(f"Grid search: {flow_name}")
        print(f"Parameters: {param_names}")
        print(
            f"Combinations: {len(combinations)} × {runs_per_combination} runs = {total_experiments} experiments"
        )
        print(f"Output: {self.base_output_dir}")
        print()

        experiment_summary = {
            "start_time": datetime.now().isoformat(),
            "flow_name": flow_name,
            "parameter_grid": parameter_grid,
            "total_experiments": total_experiments,
            "results": [],
        }

        experiment_id = 1
        for combination in combinations:
            parameters = dict(zip(param_names, combination, strict=False))

            for run in range(runs_per_combination):
                print(f"Exp {experiment_id:03d}: {parameters}")
                    
                try:
                    result = self.run_single_experiment(
                        experiment_id=experiment_id,
                        flow_name=flow_name,
                        parameters=parameters,
                        input_path=input_path,
                        external_files=external_files,
                    )
                    experiment_summary["results"].append(result)
                    print("  ✓ Completed")

                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    error_result = {
                        "experiment_id": experiment_id,
                        "flow_name": flow_name,
                        "parameters": parameters,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    experiment_summary["results"].append(error_result)

                experiment_id += 1

        # Save summary
        experiment_summary["end_time"] = datetime.now().isoformat()
        summary_file = self.base_output_dir / "grid_search_summary.json"
        with open(summary_file, "w") as f:
            json.dump(experiment_summary, f, indent=2)

        print(f"\nGrid search completed: {summary_file}")
        return experiment_summary["results"]


def main():
    """Main function to run the grid search."""
    import argparse

    parser = argparse.ArgumentParser(description="Run grid search experiments")
    parser.add_argument("input_path", type=Path, help="Path to input files/directory")
    parser.add_argument("flow_name", help="Name of the flow to use")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/vigji/Desktop/pages_sample-data/concrete_old/grid_search"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of runs per parameter combination"
    )
    parser.add_argument("--suppliers-csv", type=Path, help="Path to suppliers CSV file")
    parser.add_argument("--prices-csv", type=Path, help="Path to prices CSV file")
    parser.add_argument("--fattura", type=Path, help="Path to fattura XML file")

    # Dynamic parameter arguments
    parser.add_argument(
        "--param",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUES"),
        help="Parameter to vary: --param name 'value1,value2,value3'",
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Error: Input path does not exist: {args.input_path}")
        return 1

    # Parse parameter grid
    parameter_grid = {}
    if args.param:
        for param_name, param_values_str in args.param:
            # Try to parse as different types
            values = []
            for val_str in param_values_str.split(","):
                val_str = val_str.strip()
                # Try int, float, then string
                try:
                    if "." in val_str:
                        values.append(float(val_str))
                    else:
                        values.append(int(val_str))
                except ValueError:
                    values.append(val_str)
            parameter_grid[param_name] = values

    # Prepare external files
    external_files = {}
    if args.suppliers_csv and args.suppliers_csv.exists():
        external_files["suppliers_csv"] = args.suppliers_csv
    if args.prices_csv and args.prices_csv.exists():
        external_files["prices_csv"] = args.prices_csv
    if args.fattura and args.fattura.exists():
        external_files["fattura"] = args.fattura


    # Create and run grid search
    runner = GridSearchRunner(args.output_dir)

    try:
        results = runner.run_grid_search(
            input_path=args.input_path,
            flow_name=args.flow_name,
            parameter_grid=parameter_grid,
            runs_per_combination=args.runs,
            external_files=external_files if external_files else None,
        )

        successful = len([r for r in results if "error" not in r])
        failed = len([r for r in results if "error" in r])
        print(f"Results: {successful} successful, {failed} failed")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
