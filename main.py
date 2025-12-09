#!/usr/bin/env python3
"""
main.py

Main orchestrator for Transformer LLM Assignment.
Runs all 5 tasks in sequence with proper error handling and progress tracking.

Usage:
    python main.py                     # Run all tasks
    python main.py --tasks 1 2 3       # Run specific tasks
    python main.py --skip-heavy        # Skip GPU-intensive tasks (4, 5)
    python main.py --validate-only     # Only validate environment
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


# -------------------------
# Task Definitions
# -------------------------

TASKS = {
    1: {
        'name': 'Transformer Classifier',
        'script': 'transformer_classifier.py',
        'description': 'Custom Transformer encoder for news impact classification',
        'args': [
            '--csv_path', './datasets/vectorized_news_skipgram_embeddings.csv',
            '--epochs', '10',
            '--batch_size', '32',
            '--save_path', './outputs/transformer_classifier.pth'
        ],
        'prereqs': ['./datasets/vectorized_news_skipgram_embeddings.csv'],
        'outputs': ['./outputs/transformer_classifier.pth'],
        'gpu_heavy': False,
        'estimated_time': '5-15 minutes'
    },

    2: {
        'name': 'Extractive Summarization',
        'script': 'extractive_summary.py',
        'description': 'Sentence-similarity based extractive summarization',
        'args': [
            '--csv', './datasets/all_news.csv',
            '--index', '0',
            '--top_k', '3',
            '--out', './outputs/extractive_summary.txt'
        ],
        'prereqs': ['./datasets/all_news.csv'],
        'outputs': ['./outputs/extractive_summary.txt'],
        'gpu_heavy': False,
        'estimated_time': '1-5 minutes'
    },

    3: {
        'name': 'Abstractive Summarization',
        'script': 'abstractive_summary.py',
        'description': 'Encoder-decoder Transformer for article->headline generation',
        'args': [
            '--csv', './datasets/all_news.csv',
            '--num_pairs', '200',
            '--epochs', '10',
            '--batch_size', '32',
            '--save_path', './outputs/abstractive_transformer.pth'
        ],
        'prereqs': ['./datasets/all_news.csv'],
        'outputs': ['./outputs/abstractive_transformer.pth'],
        'gpu_heavy': False,
        'estimated_time': '10-30 minutes'
    },

    4: {
        'name': 'PEFT Fine-tuning',
        'script': 'params_finetune.py',
        'description': 'Parameter-Efficient Fine-Tuning (LoRA/Prefix/Adapter) on CS handbook',
        'args': [
            '--pdf_path', './datasets/cpsc-handbook-2022.pdf',
            '--method', 'lora',
            '--output_dir', './outputs/ft_out',
            '--epochs', '2',
            '--max_length', '1024',
            '--stride', '128'
        ],
        'prereqs': ['./datasets/cpsc-handbook-2022.pdf'],
        'outputs': ['./outputs/ft_out/finetune_answers.json'],
        'gpu_heavy': True,
        'estimated_time': '15-45 minutes (GPU) / 2-6 hours (CPU)'
    },

    5: {
        'name': 'RAG System',
        'script': 'rag_tune.py',
        'description': 'Retrieval-Augmented Generation with semantic embeddings',
        'args': [
            '--pdf_path', './datasets/cpsc-handbook-2022.pdf',
            '--output_dir', './outputs/rag_out',
            '--max_length', '1024',
            '--stride', '128',
            '--embedding_batch_size', '4',
            '--top_k', '3'
        ],
        'prereqs': ['./datasets/cpsc-handbook-2022.pdf'],
        'outputs': ['./outputs/rag_out/rag_answers.json'],
        'gpu_heavy': True,
        'estimated_time': '20-60 minutes (GPU) / 3-8 hours (CPU)'
    }
}


# -------------------------
# Helper Functions
# -------------------------

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"{text.center(70)}")
    print(f"{'='*70}\n")


def print_success(text: str):
    """Print success message"""
    print(f"[OK] {text}")


def print_error(text: str):
    """Print error message"""
    print(f"[FAIL] {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"[WARN] {text}")


def print_info(text: str):
    """Print info message"""
    print(f"[INFO] {text}")


def validate_environment() -> Dict[str, bool]:
    """
    Validate the execution environment.

    Returns:
        Dictionary of validation results
    """
    results = {}

    # Check Python version
    python_version = sys.version_info
    results['python_ok'] = python_version >= (3, 9)

    # Check required modules
    required_modules = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('peft', 'peft'),
        ('PyPDF2', 'PyPDF2'),
        ('numpy', 'numpy'),
        ('sklearn', 'sklearn')
    ]

    for display_name, import_name in required_modules:
        try:
            __import__(import_name)
            results[f'{display_name}_installed'] = True
        except ImportError:
            results[f'{display_name}_installed'] = False

    # Check GPU availability
    try:
        import torch
        results['gpu_available'] = torch.cuda.is_available()
        if results['gpu_available']:
            results['gpu_name'] = torch.cuda.get_device_name(0)
    except:
        results['gpu_available'] = False

    return results


def print_environment_status(env_results: Dict[str, bool]):
    """Print environment validation results"""
    print_header("ENVIRONMENT STATUS")

    # Python version
    if env_results.get('python_ok', False):
        print_success(f"Python version: {sys.version.split()[0]}")
    else:
        print_error(f"Python version: {sys.version.split()[0]} (requires 3.9+)")

    # Modules
    print("\nRequired Python Packages:")
    modules = ['torch', 'transformers', 'peft', 'PyPDF2', 'numpy', 'sklearn']
    for module in modules:
        if env_results.get(f'{module}_installed', False):
            print_success(f"  {module}")
        else:
            print_error(f"  {module} - NOT INSTALLED")

    # GPU
    print("\nGPU Status:")
    if env_results.get('gpu_available', False):
        print_success(f"  GPU detected: {env_results.get('gpu_name', 'Unknown')}")
    else:
        print_warning("  No GPU detected - will use CPU (much slower)")

    # Overall status
    all_ok = all([
        env_results.get('python_ok', False),
        env_results.get('torch_installed', False),
        env_results.get('transformers_installed', False),
        env_results.get('peft_installed', False),
        env_results.get('PyPDF2_installed', False),
        env_results.get('numpy_installed', False),
        env_results.get('sklearn_installed', False)
    ])

    print()
    if all_ok:
        print_success("Environment ready!")
    else:
        print_error("Environment has missing dependencies")
        print_info("Run: pip install -r requirements.txt")

    return all_ok


def validate_prerequisites(task: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that all prerequisite files exist for a task.

    Args:
        task: Task dictionary

    Returns:
        Tuple of (all_valid, missing_files)
    """
    missing = []
    for prereq in task['prereqs']:
        if not os.path.exists(prereq):
            missing.append(prereq)

    return len(missing) == 0, missing


def run_task(task_id: int, custom_args: List[str] = None) -> Tuple[bool, float, str]:
    """
    Run a single task.

    Args:
        task_id: Task number (1-5)
        custom_args: Optional custom arguments

    Returns:
        Tuple of (success, elapsed_time, message)
    """
    task = TASKS[task_id]

    print_header(f"TASK {task_id}: {task['name']}")
    print_info(f"Description: {task['description']}")
    print_info(f"Estimated time: {task['estimated_time']}")

    if task['gpu_heavy']:
        print_warning("This task is GPU-intensive and may be slow on CPU")

    # Validate prerequisites
    prereqs_ok, missing = validate_prerequisites(task)
    if not prereqs_ok:
        print_error("Missing prerequisite files:")
        for f in missing:
            print(f"  - {f}")
        return False, 0.0, "Missing prerequisites"

    print_success("Prerequisites validated")

    # Prepare command
    args = custom_args if custom_args else task['args']
    cmd = [sys.executable, task['script']] + args

    print_info(f"Running: {task['script']}")
    print()

    # Create output directory
    for output in task['outputs']:
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # Run task
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=7200  # 2 hour timeout
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print()
            print_success(f"Task {task_id} completed in {elapsed/60:.2f} minutes")

            # Verify outputs
            missing_outputs = [o for o in task['outputs'] if not os.path.exists(o)]
            if missing_outputs:
                print_warning("Some expected outputs not found:")
                for o in missing_outputs:
                    print(f"  - {o}")
            else:
                print_success("All outputs generated successfully")

            return True, elapsed, "Success"
        else:
            print()
            print_error(f"Task {task_id} failed with return code {result.returncode}")
            return False, elapsed, f"Failed with code {result.returncode}"

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print_error(f"Task {task_id} timed out after {elapsed/60:.2f} minutes")
        return False, elapsed, "Timeout"

    except Exception as e:
        elapsed = time.time() - start_time
        print_error(f"Task {task_id} failed with exception: {e}")
        return False, elapsed, str(e)


def print_summary(results: Dict[int, Tuple[bool, float, str]]):
    """
    Print execution summary.

    Args:
        results: Dictionary mapping task_id to (success, time, message)
    """
    print_header("EXECUTION SUMMARY")

    total_time = sum(r[1] for r in results.values())
    successful = sum(1 for r in results.values() if r[0])

    print(f"Total tasks attempted: {len(results)}")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print()

    print("Task Results:")
    for task_id in sorted(results.keys()):
        success, elapsed, msg = results[task_id]
        task = TASKS[task_id]
        status = "[OK]" if success else "[FAIL]"

        print(f"{status} Task {task_id}: {task['name']}")
        print(f"    Time: {elapsed/60:.2f} minutes")
        print(f"    Status: {msg}")

        if success:
            print(f"    Outputs:")
            for output in task['outputs']:
                if os.path.exists(output):
                    size = os.path.getsize(output) / 1024
                    print(f"      - {output} ({size:.2f} KB)")
        print()


# -------------------------
# Main Function
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transformer LLM Assignment - Main Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Run all tasks
  python main.py --tasks 1 2 3       # Run tasks 1, 2, and 3
  python main.py --skip-heavy        # Skip GPU-intensive tasks (4, 5)
  python main.py --validate-only     # Only validate environment
        """
    )

    parser.add_argument(
        '--tasks',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4, 5],
        help='Specific tasks to run (default: all)'
    )

    parser.add_argument(
        '--skip-heavy',
        action='store_true',
        help='Skip GPU-intensive tasks (4 and 5)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate environment and prerequisites'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue executing tasks even if one fails'
    )

    args = parser.parse_args()

    # Print banner
    print_header("TRANSFORMER LLM ASSIGNMENT")
    print_info("Main Orchestrator for All Assignment Tasks")
    print()

    # Validate environment
    env_results = validate_environment()
    env_ok = print_environment_status(env_results)

    if args.validate_only:
        sys.exit(0 if env_ok else 1)

    if not env_ok:
        print_error("\nEnvironment validation failed. Please install missing dependencies.")
        print_info("Run: pip install -r requirements.txt")
        sys.exit(1)

    # Determine which tasks to run
    if args.tasks:
        tasks_to_run = args.tasks
    elif args.skip_heavy:
        tasks_to_run = [1, 2, 3]
    else:
        tasks_to_run = [1, 2, 3, 4, 5]

    print()
    print_info(f"Tasks to run: {tasks_to_run}")
    print()

    # Warn about GPU-intensive tasks on CPU
    if not env_results.get('gpu_available', False):
        heavy_tasks = [t for t in tasks_to_run if TASKS[t]['gpu_heavy']]
        if heavy_tasks:
            print_warning("GPU-intensive tasks detected but no GPU available:")
            for t in heavy_tasks:
                print(f"  - Task {t}: {TASKS[t]['name']}")
            print_warning("These tasks will run on CPU and may be VERY slow")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting.")
                sys.exit(0)

    # Create outputs directory
    os.makedirs("./outputs", exist_ok=True)

    # Run tasks
    results = {}
    overall_start = time.time()

    for task_id in tasks_to_run:
        success, elapsed, msg = run_task(task_id)
        results[task_id] = (success, elapsed, msg)

        if not success and not args.continue_on_error:
            print_error("\nTask failed. Stopping execution.")
            print_info("Use --continue-on-error to continue on failures")
            break

    overall_elapsed = time.time() - overall_start

    # Print summary
    print_summary(results)

    # Final message
    all_success = all(r[0] for r in results.values())
    if all_success:
        print_header("ALL TASKS COMPLETED SUCCESSFULLY!")
        print_info(f"Total execution time: {overall_elapsed/60:.2f} minutes")
        print()
        print_info("Next steps:")
        print("  1. Review outputs in ./outputs/ directory")
        print("  2. Compare results for assignment report")
        print("  3. Analyze model performance and quality")
    else:
        print_header("EXECUTION COMPLETED WITH ERRORS")
        failed_tasks = [t for t, r in results.items() if not r[0]]
        print_error(f"Failed tasks: {failed_tasks}")

    # Save execution log
    log_path = "./outputs/execution_log.json"
    with open(log_path, 'w') as f:
        log_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': {k: str(v) for k, v in env_results.items()},
            'tasks_run': tasks_to_run,
            'results': {
                str(k): {
                    'success': v[0],
                    'elapsed_seconds': v[1],
                    'message': v[2]
                }
                for k, v in results.items()
            },
            'total_time_seconds': overall_elapsed
        }
        json.dump(log_data, f, indent=2)

    print_info(f"Execution log saved to {log_path}")

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
