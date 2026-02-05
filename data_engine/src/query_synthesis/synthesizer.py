"""
Query Synthesizer - Main orchestration for question-answer synthesis.

This module provides the main interface for synthesizing question-answer
pairs from document data.
"""

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import random

import jsonlines
import ray
import transformers
from PIL import Image
from tqdm import tqdm

from ..core.model_client import LMClient
from ..core.utils import load_config, pretty_name, convert_pdf_to_image
from .anchors import ImageAnchor, TableAnchor, TextAnchor, set_model_references
from .document_parser import DocumentParser


@ray.remote
def process_anchor(
    anchor: Union[ImageAnchor, TableAnchor, TextAnchor],
    output_dir: str,
    debug: bool = False
) -> bool:
    """
    Process a single anchor to generate questions.
    
    This is a Ray remote function for parallel processing.
    
    Args:
        anchor: The anchor object to process.
        output_dir: Directory to save output files.
        debug: Whether to print debug information.
    
    Returns:
        True if processing was successful.
    """
    anchor_name = pretty_name(anchor.name)
    file_id = pretty_name(anchor.file_id)
    output_file = os.path.join(output_dir, f"{file_id}_{anchor_name}_questions.jsonl")
    
    # Skip if already processed
    if os.path.exists(output_file):
        print(f"[Synthesizer] Skipping existing: {output_file}")
        return True
    
    try:
        if isinstance(anchor, ImageAnchor):
            # Extract key points first
            anchor.extract_key_points()
            if debug:
                print(f"[Synthesizer] Key points for {anchor.name}: {anchor.get_keypoints()}")
            
            # Generate easy questions if subtitle exists
            if anchor.subtitle:
                anchor.synthesize_question_v0(synthesize_role='easy')
            
            # Generate hard questions if related text exists
            if anchor.related_text:
                anchor.synthesize_question_v1(synthesize_role='hard')
        
        elif isinstance(anchor, TableAnchor):
            # Generate questions for different abilities
            abilities = ['extraction', 'calculation', 'comparison']
            if anchor.related_text:
                abilities.append('analysis')
            
            for ability in abilities:
                anchor.synthesize_question(ability)
        
        elif isinstance(anchor, TextAnchor):
            anchor.synthesize_question()
        
        else:
            raise TypeError(f"Unsupported anchor type: {type(anchor)}")
        
        # Save results
        results = anchor.get_results()
        if not results:
            print(f"[Synthesizer] No questions generated for {anchor.name}")
            return False
        
        with jsonlines.open(output_file, mode='w') as writer:
            for question in results:
                writer.write(question)
        
        print(f"[Synthesizer] Saved {len(results)} questions to {output_file}")
        return True
    
    except Exception as e:
        print(f"[Synthesizer] Error processing {anchor.name}: {e}")
        return False


class QuerySynthesizer:
    """
    Main class for synthesizing question-answer pairs from documents.
    
    This class orchestrates the entire query synthesis pipeline:
    1. Parse document PDFs to extract figures, tables, and text
    2. Create anchors for each element
    3. Generate question-answer pairs using an LLM
    4. Save results to JSONL files
    
    Args:
        config_path: Path to the YAML configuration file.
        port: Port for the vLLM server.
        num_cpus: Number of CPUs for Ray parallelization.
    
    Example:
        >>> synthesizer = QuerySynthesizer(
        ...     config_path="configs/query_synthesis.yaml",
        ...     port=8000
        ... )
        >>> synthesizer.run(
        ...     process_figures=True,
        ...     process_tables=True,
        ...     process_text=False
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
        
        self.image_output_dir = self.config.get('output', {}).get(
            'image_dir', 
            os.path.join(self.output_dir, 'images')
        )
        os.makedirs(self.image_output_dir, exist_ok=True)
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the language model and tokenizer."""
        print("[Synthesizer] Loading model and tokenizer...")
        
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
        
        # Set global references for anchor classes
        set_model_references(self.tokenizer, self.model)
        
        print(f"[Synthesizer] Model loaded: {model_config['model_name']}")
    
    def find_json_files(self) -> List[str]:
        """
        Find all valid JSON files (with corresponding PDFs) in the input directory.
        
        Returns:
            List of JSON file paths.
        """
        json_files = []
        raw_data_path = self.config['input']['raw_data_path']
        max_files = self.config['input'].get('max_num_files', float('inf'))
        
        for root, dirs, files in os.walk(raw_data_path):
            has_pdf = any(f.lower().endswith('.pdf') for f in files)
            json_file = next((f for f in files if f.lower().endswith('.json')), None)
            
            if json_file and has_pdf:
                json_files.append(os.path.join(root, json_file))
        
        if len(json_files) > max_files:
            print(f"[Synthesizer] Found {len(json_files)} files, limiting to {max_files}")
            json_files = json_files[:max_files]
        
        print(f"[Synthesizer] Found {len(json_files)} valid JSON files")
        return json_files
    
    def create_anchors_from_document(
        self,
        json_path: str,
        process_figures: bool = True,
        process_tables: bool = True,
        process_text: bool = False
    ) -> List[Union[ImageAnchor, TableAnchor, TextAnchor]]:
        """
        Create anchors from a single document.
        
        Args:
            json_path: Path to the document JSON file.
            process_figures: Whether to create figure anchors.
            process_tables: Whether to create table anchors.
            process_text: Whether to create text anchors.
        
        Returns:
            List of anchor objects.
        """
        anchors = []
        
        # Parse document
        parser = DocumentParser(json_path, self.image_output_dir)
        figures, tables, texts = parser.parse()
        
        # Process figures
        if process_figures and figures:
            merged_titles = parser.merge_figure_titles()
            
            # Check if we have enough titles
            if len(figures) > len(merged_titles):
                print(f"[Synthesizer] Warning: More figures ({len(figures)}) than titles ({len(merged_titles)}) in {json_path}")
            else:
                sorted_figures = parser.sort_figures()
                
                for idx, fig_info in enumerate(sorted_figures):
                    subtitle, related_text, related_pages, hints2bbox = \
                        parser.get_figure_references(idx + 1)
                    
                    if not subtitle and not related_text:
                        continue
                    
                    hints2bbox[f"Figure {idx + 1}"] = fig_info['bbox']
                    
                    try:
                        anchor = ImageAnchor(
                            file_id=parser.file_id,
                            name=f"Figure {idx + 1}",
                            pdf_path=parser.pdf_path,
                            image=Image.open(fig_info['image_path']),
                            image_path=fig_info['image_path'],
                            subtitle=subtitle,
                            related_text=related_text,
                            related_pages=related_pages,
                            hints2bbox=hints2bbox
                        )
                        anchors.append(anchor)
                    except Exception as e:
                        print(f"[Synthesizer] Error creating figure anchor: {e}")
        
        # Process tables
        if process_tables and tables:
            existing_table_numbers = set()
            
            for table in tables:
                idx = table['node_idx']
                
                # Find table title from adjacent nodes
                nodes = parser._nodes
                subtitle = ""
                subtitle_bbox = None
                
                if idx > 0 and nodes[idx - 1].get('tabletitle', False):
                    subtitle = nodes[idx - 1].get('content', '').strip()
                    subtitle_bbox = nodes[idx - 1].get('locs', [{}])[0].get('bbox')
                elif idx + 1 < len(nodes) and nodes[idx + 1].get('tabletitle', False):
                    subtitle = nodes[idx + 1].get('content', '').strip()
                    subtitle_bbox = nodes[idx + 1].get('locs', [{}])[0].get('bbox')
                
                if not subtitle:
                    continue
                
                # Extract table number
                import re
                match = re.search(r"Table\s+(\d+)", subtitle, re.IGNORECASE)
                if not match:
                    continue
                
                table_number = match.group(1)
                if table_number in existing_table_numbers:
                    continue
                
                existing_table_numbers.add(table_number)
                
                # Get related text
                _, related_text, related_pages, hints2bbox = \
                    parser.get_table_references(table_number)
                
                hints2bbox[subtitle] = subtitle_bbox
                hints2bbox[f"Table {table_number}"] = table['bbox']
                
                if table['page'] not in related_pages:
                    related_pages.append(table['page'])
                
                try:
                    anchor = TableAnchor(
                        file_id=parser.file_id,
                        name=f"Table {table_number}",
                        pdf_path=parser.pdf_path,
                        image=Image.open(table['image_path']),
                        image_path=table['image_path'],
                        markdown_content=table['md_content'],
                        subtitle=subtitle,
                        related_text=related_text,
                        related_pages=related_pages,
                        hints2bbox=hints2bbox
                    )
                    anchors.append(anchor)
                except Exception as e:
                    print(f"[Synthesizer] Error creating table anchor: {e}")
        
        # Process text
        if process_text and texts:
            # Group text by page
            page_texts = defaultdict(list)
            page_bboxes = defaultdict(dict)
            
            for content, page, bbox in texts:
                page_texts[page].append(content)
                page_bboxes[page][content] = bbox
            
            # Sample pages if too many
            pages = list(page_texts.keys())
            if len(pages) > 10:
                pages = random.sample(pages, 10)
            
            for page in pages:
                related_text = "\n\n".join(page_texts[page])
                if not related_text.strip():
                    continue
                
                # Create page image
                image_path = os.path.join(
                    self.image_output_dir,
                    parser.file_id,
                    f"page_{page}.png"
                )
                
                try:
                    if not os.path.exists(image_path):
                        page_image = convert_pdf_to_image(parser.pdf_path, page)
                        page_image.save(image_path)
                    
                    anchor = TextAnchor(
                        file_id=parser.file_id,
                        name=f"Page {page}",
                        pdf_path=parser.pdf_path,
                        image=Image.open(image_path),
                        image_path=image_path,
                        related_text=related_text,
                        related_pages=[page],
                        hints2bbox=page_bboxes[page]
                    )
                    anchors.append(anchor)
                except Exception as e:
                    print(f"[Synthesizer] Error creating text anchor: {e}")
        
        return anchors
    
    def run(
        self,
        process_figures: bool = True,
        process_tables: bool = True,
        process_text: bool = False,
        debug: bool = False
    ):
        """
        Run the query synthesis pipeline.
        
        Args:
            process_figures: Whether to process figure elements.
            process_tables: Whether to process table elements.
            process_text: Whether to process pure text elements.
            debug: Whether to print debug information.
        """
        # Find input files
        json_files = self.find_json_files()
        
        # Create anchors from all documents
        all_anchors = []
        print("[Synthesizer] Creating anchors from documents...")
        
        for json_path in tqdm(json_files, desc="Parsing documents"):
            anchors = self.create_anchors_from_document(
                json_path,
                process_figures=process_figures,
                process_tables=process_tables,
                process_text=process_text
            )
            all_anchors.extend(anchors)
        
        print(f"[Synthesizer] Created {len(all_anchors)} anchors")
        
        if not all_anchors:
            print("[Synthesizer] No anchors to process")
            return
        
        # Initialize Ray and process anchors in parallel
        start_time = time.time()
        ray.init(num_cpus=self.num_cpus)
        
        try:
            tasks = [
                process_anchor.remote(anchor, self.output_dir, debug)
                for anchor in all_anchors
            ]
            
            # Track progress
            with tqdm(total=len(tasks), desc="Synthesizing questions") as pbar:
                while tasks:
                    ready, tasks = ray.wait(tasks)
                    pbar.update(len(ready))
            
            elapsed = time.time() - start_time
            print(f"[Synthesizer] Completed in {elapsed:.2f} seconds")
            print(f"[Synthesizer] Results saved to {self.output_dir}")
        
        finally:
            ray.shutdown()


def main():
    """Command-line interface for query synthesis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Synthesize question-answer pairs from documents"
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
        help="Process pure text"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Default to processing figures if nothing specified
    if not args.figure and not args.table and not args.pure_text:
        args.figure = True
    
    synthesizer = QuerySynthesizer(
        config_path=args.config_file,
        port=args.port,
        num_cpus=args.num_cpus
    )
    
    synthesizer.run(
        process_figures=args.figure,
        process_tables=args.table,
        process_text=args.pure_text,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
