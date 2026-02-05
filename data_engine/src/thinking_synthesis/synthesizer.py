"""
Thinking Pattern Synthesizer - Main orchestration for thinking synthesis.

This module provides the main interface for synthesizing thinking patterns
(chain-of-thought reasoning) from question-answer pairs.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Union

import jsonlines
import ray
import transformers
from PIL import Image
from tqdm import tqdm

from ..core.model_client import LMClient
from ..core.utils import load_config, convert_pdf_to_image, image_to_base64
from .processor import RefinementProcessor, PosteriorManager


@ray.remote
def process_item(
    item: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    output_dir: str,
    mode: str = "initial",
    use_refinement: bool = True,
    max_pages: int = 20,
    debug: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Process a single item to generate thinking pattern.
    
    Args:
        item: Dictionary containing question, answer, and page information.
        model: The language model client.
        tokenizer: The tokenizer.
        output_dir: Directory to save output.
        mode: Either 'initial' or 'retrieval'.
        use_refinement: Whether to use iterative refinement.
        max_pages: Maximum number of pages to process.
        debug: Whether to print debug information.
    
    Returns:
        Processed item with thinking pattern, or None on failure.
    """
    try:
        # Load page images
        pdf_path = item.get("pdf_path", "")
        pages = item.get("related_pages", item.get("pages", []))
        
        if not pdf_path or not pages:
            print(f"[ThinkingSynthesizer] Missing pdf_path or pages")
            return None
        
        # Limit pages
        if len(pages) > max_pages:
            pages = pages[:max_pages]
        
        images = []
        for page_num in pages:
            try:
                page_image = convert_pdf_to_image(pdf_path, page_num)
                images.append(page_image)
            except Exception as e:
                print(f"[ThinkingSynthesizer] Error loading page {page_num}: {e}")
                continue
        
        if not images:
            print(f"[ThinkingSynthesizer] No images loaded")
            return None
        
        question = item.get("question", "")
        ref_answer = item.get("answer", "")
        context = item.get("context", None)
        
        if not question or not ref_answer:
            print(f"[ThinkingSynthesizer] Missing question or answer")
            return None
        
        # Create processor
        processor = RefinementProcessor(
            model=model,
            tokenizer=tokenizer,
            max_iterations=3 if use_refinement else 0,
            use_judge=use_refinement
        )
        
        if use_refinement:
            result = processor.refine(question, images, ref_answer, context)
        else:
            initial = processor.generate_initial(question, images, context)
            result = initial[0] if initial else {"analysis": "", "answer": ""}
        
        # Build output item
        output_item = {
            **item,
            "thinking": result.get("analysis", ""),
            "generated_answer": result.get("answer", ""),
        }
        
        if "iterations" in result:
            output_item["refinement_iterations"] = len(result["iterations"])
        
        if debug:
            print(f"[ThinkingSynthesizer] Generated thinking for: {question[:50]}...")
        
        return output_item
    
    except Exception as e:
        print(f"[ThinkingSynthesizer] Error processing item: {e}")
        return None


class ThinkingSynthesizer:
    """
    Main class for synthesizing thinking patterns from QA pairs.
    
    This class orchestrates the thinking pattern synthesis pipeline:
    1. Load question-answer pairs from JSONL files
    2. Load corresponding document page images
    3. Generate thinking patterns using an LLM
    4. Optionally refine patterns using reference answers
    5. Save results to JSONL files
    
    Args:
        config_path: Path to the YAML configuration file.
        port: Port for the vLLM server.
        num_cpus: Number of CPUs for Ray parallelization.
    
    Example:
        >>> synthesizer = ThinkingSynthesizer(
        ...     config_path="configs/thinking_synthesis.yaml",
        ...     port=8000
        ... )
        >>> synthesizer.run(
        ...     mode="initial",
        ...     use_refinement=True
        ... )
    """
    
    def __init__(
        self,
        config_path: str,
        port: int = 8000,
        num_cpus: int = 32
    ):
        self.config = load_config(config_path)
        self.port = port
        self.num_cpus = num_cpus
        
        # Set up paths
        self.output_dir = self.config['output']['file_prefix']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the language model and tokenizer."""
        print("[ThinkingSynthesizer] Loading model and tokenizer...")
        
        model_config = self.config["model"]
        model_args = model_config.get("model_args", {})
        model_args["port"] = self.port
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_config["model_name"],
            use_fast=True,
            padding_side="right",
            truncation_side="right",
        )
        
        self.model = LMClient(
            model_type=model_config["model_type"],
            model_name=model_config["model_name"],
            num_rollouts=1,
            tokenizer=self.tokenizer,
            **model_args
        )
        
        print(f"[ThinkingSynthesizer] Model loaded: {model_config['model_name']}")
    
    def load_input_data(self) -> List[Dict[str, Any]]:
        """
        Load input data from JSONL files.
        
        Returns:
            List of input items.
        """
        input_path = self.config['input']['file_path']
        items = []
        
        if os.path.isfile(input_path):
            files = [input_path]
        else:
            files = [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.endswith('.jsonl')
            ]
        
        max_items = self.config['input'].get('max_num_items', float('inf'))
        
        for file_path in files:
            try:
                with jsonlines.open(file_path) as reader:
                    for item in reader:
                        items.append(item)
                        if len(items) >= max_items:
                            break
            except Exception as e:
                print(f"[ThinkingSynthesizer] Error loading {file_path}: {e}")
            
            if len(items) >= max_items:
                break
        
        print(f"[ThinkingSynthesizer] Loaded {len(items)} items")
        return items
    
    def run(
        self,
        mode: str = "initial",
        use_refinement: bool = True,
        max_pages: int = 20,
        debug: bool = False
    ):
        """
        Run the thinking synthesis pipeline.
        
        Args:
            mode: Either 'initial' (standard) or 'retrieval' (with page selection).
            use_refinement: Whether to iteratively refine thinking patterns.
            max_pages: Maximum number of pages to include per document.
            debug: Whether to print debug information.
        """
        # Load input data
        items = self.load_input_data()
        
        if not items:
            print("[ThinkingSynthesizer] No input data found")
            return
        
        # Output file
        output_file = os.path.join(
            self.output_dir,
            f"thinking_{mode}{'_refined' if use_refinement else ''}.jsonl"
        )
        
        # Check for already processed items
        processed_ids = set()
        if os.path.exists(output_file):
            with jsonlines.open(output_file) as reader:
                for item in reader:
                    item_id = item.get("id", item.get("question", ""))
                    processed_ids.add(item_id)
        
        # Filter out already processed
        items_to_process = []
        for item in items:
            item_id = item.get("id", item.get("question", ""))
            if item_id not in processed_ids:
                items_to_process.append(item)
        
        print(f"[ThinkingSynthesizer] Processing {len(items_to_process)} items "
              f"({len(processed_ids)} already done)")
        
        if not items_to_process:
            print("[ThinkingSynthesizer] All items already processed")
            return
        
        # Initialize Ray
        start_time = time.time()
        ray.init(num_cpus=self.num_cpus)
        
        try:
            # Submit tasks
            tasks = [
                process_item.remote(
                    item=item,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    output_dir=self.output_dir,
                    mode=mode,
                    use_refinement=use_refinement,
                    max_pages=max_pages,
                    debug=debug
                )
                for item in items_to_process
            ]
            
            # Process results and save incrementally
            with jsonlines.open(output_file, mode='a') as writer:
                with tqdm(total=len(tasks), desc="Synthesizing thinking") as pbar:
                    while tasks:
                        ready, tasks = ray.wait(tasks, timeout=10.0)
                        
                        for obj in ready:
                            result = ray.get(obj)
                            if result:
                                writer.write(result)
                            pbar.update(1)
            
            elapsed = time.time() - start_time
            print(f"[ThinkingSynthesizer] Completed in {elapsed:.2f} seconds")
            print(f"[ThinkingSynthesizer] Results saved to {output_file}")
        
        finally:
            ray.shutdown()
    
    def run_batch_refinement(
        self,
        input_file: str,
        output_file: str,
        num_candidates: int = 4
    ):
        """
        Run batch refinement with multiple candidates per question.
        
        This method generates multiple thinking pattern candidates for
        each question and selects the best one.
        
        Args:
            input_file: Path to input JSONL file.
            output_file: Path to output JSONL file.
            num_candidates: Number of candidates per question.
        """
        # Load input items
        items = []
        with jsonlines.open(input_file) as reader:
            for item in reader:
                items.append(item)
        
        print(f"[ThinkingSynthesizer] Processing {len(items)} items with "
              f"{num_candidates} candidates each")
        
        # Initialize manager
        manager = PosteriorManager(
            model=self.model,
            tokenizer=self.tokenizer,
            num_candidates=num_candidates
        )
        
        # Process items
        results = []
        for item in tqdm(items, desc="Batch refinement"):
            # Load images
            pdf_path = item.get("pdf_path", "")
            pages = item.get("related_pages", item.get("pages", []))
            
            images = []
            for page_num in pages[:20]:  # Limit pages
                try:
                    page_image = convert_pdf_to_image(pdf_path, page_num)
                    images.append(page_image)
                except:
                    continue
            
            if not images:
                continue
            
            # Get best candidate
            best = manager.select_best(
                question=item.get("question", ""),
                images=images,
                ref_answer=item.get("answer", ""),
                context=item.get("context")
            )
            
            result = {
                **item,
                "thinking": best.get("analysis", ""),
                "generated_answer": best.get("answer", ""),
                "score": best.get("score", 0)
            }
            results.append(result)
        
        # Save results
        with jsonlines.open(output_file, mode='w') as writer:
            for result in results:
                writer.write(result)
        
        print(f"[ThinkingSynthesizer] Saved {len(results)} results to {output_file}")


def main():
    """Command-line interface for thinking synthesis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Synthesize thinking patterns from QA pairs"
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
        help="Port for vLLM server"
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=32,
        help="Number of CPUs for Ray"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="initial",
        choices=["initial", "retrieval"],
        help="Synthesis mode"
    )
    parser.add_argument(
        "--no-refinement",
        action="store_true",
        help="Disable iterative refinement"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum pages per document"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
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
