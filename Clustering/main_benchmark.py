import argparse
import sys
import importlib.util
import os

def dynamic_import(directory, module_name):
    """
    Dynamically imports a module from a specific directory.
    
    Args:
        directory (str): The path to the directory containing the module.
        module_name (str): The name of the module (without .py extension).
    
    Returns:
        module: The imported module.
    """
    module_path = os.path.join(directory, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    # Step 1: Create ArgumentParser
    parser = argparse.ArgumentParser(
        description="Run clustering benchmarks on a specified dataset using Aeon or Tslearn."
    )

    # Step 2: Add arguments
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset to benchmark. Must be one of the predefined datasets.",
    )
    parser.add_argument(
        "--library",
        choices=["aeon", "tslearn"],
        required=True,
        help="Library to use for clustering. Must be either 'aeon' or 'tslearn'.",
    )

    # Step 3: Parse arguments
    args = parser.parse_args()

    # Step 4: Determine implementation directory
    if args.library == "aeon":
        implementation_dir = "aeon_implementations"
    elif args.library == "tslearn":
        implementation_dir = "tslearn_implementations"
    
    # Step 5: Dynamically import modules from the chosen implementation directory
    try:
        data_loader = dynamic_import(implementation_dir, "data_loader")
        clustering_methods = dynamic_import(implementation_dir, "clustering_methods")
        preprocessor = dynamic_import(implementation_dir, "preprocessor")
        evaluator = dynamic_import(implementation_dir, "evaluator")
        
        # Import run_benchmark function from benchmark.py in the chosen directory
        benchmark_module = dynamic_import(implementation_dir, "benchmark")
        
        # Validate dataset name
        if args.dataset_name not in data_loader.working_datasets:
            print(f"Error: Dataset '{args.dataset_name}' is not in the list of available datasets.")
            print(f"Available datasets: {', '.join(data_loader.working_datasets)}")
            sys.exit(1)

        # Run the benchmark
        print(f"Running benchmark for dataset '{args.dataset_name}' using '{args.library}' clustering methods.")
        results_df, clustering_outputs = benchmark_module.run_benchmark(args.dataset_name)

        # Save results or process further as needed
        print("Benchmark completed successfully.")

    except Exception as e:
        print(f"An error occurred while running the benchmark: {e}")
        sys.exit(1)
        
#############################################################################################################################
        
if __name__ == "__main__":
    main()
