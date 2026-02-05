#!/usr/bin/env python3
"""
Run Query Synthesis Pipeline

This script runs the query synthesis pipeline to generate
question-answer pairs from parsed PDF documents.

Usage:
    python run_query_synthesis.py --config-file configs/query_synthesis.yaml --port 8000
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_synthesis import QuerySynthesizer


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize question-answer pairs from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process figures only
    python run_query_synthesis.py --config-file configs/query_synthesis.yaml --port 8000 --figure

    # Process all element types
    python run_query_synthesis.py --config-file configs/query_synthesis.yaml --port 8000 --figure --table --pure-text

    # Debug mode with fewer CPUs
    python run_query_synthesis.py --config-file configs/query_synthesis.yaml --port 8000 --num-cpus 4 --debug
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
        "--figure",
        action="store_true",
        help="Process figures"
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Process tables"
    )
    parser.add_argument(
        "--pure-text",
        action="store_true",
        help="Process pure text elements"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Default to processing figures if nothing specified
    process_figures = args.figure
    process_tables = args.table
    process_text = args.pure_text
    
    if not (process_figures or process_tables or process_text):
        print("No element types specified, defaulting to figures only.")
        process_figures = True
    
    print(f"Configuration: {args.config_file}")
    print(f"vLLM Port: {args.port}")
    print(f"CPUs: {args.num_cpus}")
    print(f"Processing: figures={process_figures}, tables={process_tables}, text={process_text}")
    print()
    
    # Run synthesizer
    synthesizer = QuerySynthesizer(
        config_path=args.config_file,
        port=args.port,
        num_cpus=args.num_cpus
    )
    
    synthesizer.run(
        process_figures=process_figures,
        process_tables=process_tables,
        process_text=process_text,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
