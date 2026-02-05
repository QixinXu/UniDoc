#!/usr/bin/env python3
"""
Run Thinking Pattern Synthesis Pipeline

This script runs the thinking pattern synthesis pipeline to generate
chain-of-thought reasoning from question-answer pairs.

Usage:
    python run_thinking_synthesis.py --config-file configs/thinking_synthesis.yaml --port 8000
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thinking_synthesis import ThinkingSynthesizer


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize thinking patterns from QA pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard mode with refinement
    python run_thinking_synthesis.py --config-file configs/thinking_synthesis.yaml --port 8000

    # Fast mode without refinement
    python run_thinking_synthesis.py --config-file configs/thinking_synthesis.yaml --port 8000 --no-refinement

    # Retrieval mode (with page selection)
    python run_thinking_synthesis.py --config-file configs/thinking_synthesis.yaml --port 8000 --mode retrieval

    # Debug mode
    python run_thinking_synthesis.py --config-file configs/thinking_synthesis.yaml --port 8000 --debug
        """
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for vLLM server (default: 8000)"
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=32,
        help="Number of CPUs for Ray parallelization (default: 32)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="initial",
        choices=["initial", "retrieval"],
        help="Synthesis mode: 'initial' for standard, 'retrieval' for page selection (default: initial)"
    )
    parser.add_argument(
        "--no-refinement",
        action="store_true",
        help="Disable iterative refinement (faster but lower quality)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum number of pages per document (default: 20)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    print(f"Configuration: {args.config_file}")
    print(f"vLLM Port: {args.port}")
    print(f"CPUs: {args.num_cpus}")
    print(f"Mode: {args.mode}")
    print(f"Refinement: {'disabled' if args.no_refinement else 'enabled'}")
    print(f"Max pages: {args.max_pages}")
    print()
    
    # Run synthesizer
    synthesizer = ThinkingSynthesizer(
        config_path=args.config_file,
        port=args.port,
        num_cpus=args.num_cpus
    )
    
    synthesizer.run(
        mode=args.mode,
        use_refinement=not args.no_refinement,
        max_pages=args.max_pages,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
