"""
Document parser for extracting structured elements from PDFs.

This module handles the parsing of document JSON files (from PDF parsing tools)
and extracts figures, tables, and text elements for question synthesis.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..core.utils import convert_pdf_to_image, crop_image, pretty_name


class DocumentParser:
    """
    Parser for extracting structured elements from parsed PDF documents.
    
    This class takes JSON files produced by PDF parsing tools and extracts
    figures, tables, and text elements along with their metadata.
    
    Args:
        json_path: Path to the JSON file containing parsed PDF structure.
        image_output_dir: Directory to save extracted images.
    
    Example:
        >>> parser = DocumentParser(
        ...     json_path="/path/to/document.json",
        ...     image_output_dir="/path/to/images"
        ... )
        >>> figures, tables, texts = parser.parse()
    """
    
    # Common patterns for referencing figures in text
    FIGURE_PREFIXES = ['Fig. ', 'Figure ', 'figure ', 'Figure', 'figure']
    FIGURE_SUFFIXES = [' ', ',', '.', ':', '?']
    
    # Common patterns for referencing tables in text
    TABLE_PREFIXES = ['Table ', 'table ']
    TABLE_SUFFIXES = [' ', ',', '.', ':', '?']
    
    def __init__(self, json_path: str, image_output_dir: str):
        self.json_path = json_path
        self.pdf_path = json_path.replace('.json', '.pdf')
        self.image_output_dir = image_output_dir
        
        # Extract file ID from path
        self.file_id = os.path.basename(json_path).replace('.json', '')
        
        # Create output directory
        self.file_image_dir = os.path.join(image_output_dir, self.file_id)
        os.makedirs(self.file_image_dir, exist_ok=True)
        
        # Storage for parsed elements
        self.figures: List[Dict] = []
        self.tables: List[Dict] = []
        self.texts: List[Tuple[str, int, List[float]]] = []  # (content, page, bbox)
        self.figure_titles: List[Tuple[str, int, List[float]]] = []  # (title, page, bbox)
        
        self._nodes: List[Dict] = []
    
    def load(self) -> bool:
        """
        Load and parse the JSON file.
        
        Returns:
            True if loading was successful, False otherwise.
        """
        try:
            with open(self.json_path, 'r') as f:
                file_content = json.load(f)
            
            data = file_content.get('data', {})
            self._nodes = data.get('nodes', [])
            
            print(f"[DocumentParser] Loaded {len(self._nodes)} nodes from {self.json_path}")
            return True
            
        except Exception as e:
            print(f"[DocumentParser] Error loading {self.json_path}: {e}")
            return False
    
    def parse(self) -> Tuple[List[Dict], List[Dict], List[Tuple]]:
        """
        Parse the document and extract all elements.
        
        Returns:
            Tuple of (figures, tables, texts).
        """
        if not self._nodes:
            if not self.load():
                return [], [], []
        
        for idx, node in enumerate(self._nodes):
            self._process_node(node, idx)
        
        return self.figures, self.tables, self.texts
    
    def _process_node(self, node: Dict, idx: int):
        """Process a single node from the document structure."""
        node_type = node.get('type', '')
        
        if node_type == 'apara':
            self._process_paragraph(node)
        elif node_type == 'afigure':
            self._process_figure(node, idx)
        elif node_type == 'atable':
            self._process_table(node, idx)
    
    def _process_paragraph(self, node: Dict):
        """Process a paragraph node."""
        is_title = node.get('texttitle', False)
        is_figure_title = node.get('figuretitle', False)
        
        try:
            page_num = node['refer']['locs'][0]['page']
            bbox = node['refer']['locs'][0]['bbox']
        except (KeyError, IndexError):
            return
        
        # Skip section titles
        if is_title:
            return
        
        # Extract content
        if 'childs' in node:
            childs_contents = []
            for child in node['childs']:
                if child.get('type') == 'aline':
                    childs_contents.append(child.get('content', ''))
            content = ' '.join(childs_contents).strip()
        else:
            content = node.get('content', '').strip()
        
        if not content:
            return
        
        if is_figure_title:
            self.figure_titles.append((content, page_num, bbox))
        else:
            self.texts.append((content, page_num, bbox))
    
    def _process_figure(self, node: Dict, idx: int):
        """Process a figure node."""
        try:
            bbox = node['refer']['locs'][0]['bbox']
            page = node['refer']['locs'][0]['page']
        except (KeyError, IndexError):
            return
        
        # Extract and save the figure image
        try:
            page_image = convert_pdf_to_image(self.pdf_path, page)
            figure = crop_image(page_image, bbox)
            
            save_path = os.path.join(
                self.file_image_dir,
                f"figure_{page}_{bbox[0]:.2f}_{bbox[1]:.2f}.png"
            )
            
            if not os.path.exists(save_path):
                figure.save(save_path)
            
            self.figures.append({
                'bbox': bbox,
                'page': page,
                'file_id': self.file_id,
                'image_path': save_path,
                'node_idx': idx
            })
            
        except Exception as e:
            print(f"[DocumentParser] Error processing figure: {e}")
    
    def _process_table(self, node: Dict, idx: int):
        """Process a table node."""
        try:
            bbox = node['refer']['locs'][0]['bbox']
            page = node['refer']['locs'][0]['page']
        except (KeyError, IndexError):
            return
        
        # Extract and save the table image
        try:
            page_image = convert_pdf_to_image(self.pdf_path, page)
            table_image = crop_image(page_image, bbox)
            
            save_path = os.path.join(
                self.file_image_dir,
                f"table_{page}_{bbox[0]:.2f}_{bbox[1]:.2f}.png"
            )
            
            if not os.path.exists(save_path):
                table_image.save(save_path)
            
            self.tables.append({
                'bbox': bbox,
                'page': page,
                'file_id': self.file_id,
                'image_path': save_path,
                'node_idx': idx,
                'md_content': node.get('content', '')
            })
            
        except Exception as e:
            print(f"[DocumentParser] Error processing table: {e}")
    
    def get_figure_references(self, figure_index: int) -> Tuple[str, str, List[int], Dict]:
        """
        Get subtitle and related text for a figure.
        
        Args:
            figure_index: 1-based index of the figure.
        
        Returns:
            Tuple of (subtitle, related_text, related_pages, hints2bbox).
        """
        anchor_subtitle = []
        anchor_related_texts = []
        anchor_related_pages = []
        hints2bbox = {}
        
        for prefix in self.FIGURE_PREFIXES:
            for suffix in self.FIGURE_SUFFIXES:
                pattern = f'{prefix}{figure_index}{suffix}'
                
                # Find matching figure titles
                for title, page, bbox in self.figure_titles:
                    if pattern in title:
                        anchor_subtitle.append(title)
                        anchor_related_pages.append(page)
                        hints2bbox[title] = bbox
                
                # Find matching text references
                for text, page, bbox in self.texts:
                    if pattern in text:
                        anchor_related_texts.append(text)
                        anchor_related_pages.append(page)
                        hints2bbox[text] = bbox
        
        subtitle = " ".join(anchor_subtitle).strip()
        related_text = "\n\n".join(t.strip() for t in anchor_related_texts if t.strip())
        related_pages = list(set(anchor_related_pages))
        
        return subtitle, related_text, related_pages, hints2bbox
    
    def get_table_references(self, table_number: str) -> Tuple[str, str, List[int], Dict]:
        """
        Get subtitle and related text for a table.
        
        Args:
            table_number: The table number (e.g., "1", "2").
        
        Returns:
            Tuple of (subtitle, related_text, related_pages, hints2bbox).
        """
        related_texts = []
        related_pages = []
        hints2bbox = {}
        
        for prefix in self.TABLE_PREFIXES:
            for suffix in self.TABLE_SUFFIXES:
                pattern = f'{prefix}{table_number}{suffix}'
                
                for text, page, bbox in self.texts:
                    if pattern in text:
                        related_texts.append(text)
                        related_pages.append(page)
                        hints2bbox[text] = bbox
        
        related_text = "\n\n".join(related_texts)
        related_pages = list(set(related_pages))
        
        return "", related_text, related_pages, hints2bbox
    
    def sort_figures(self) -> List[Dict]:
        """
        Sort figures by page and position (left-to-right, top-to-bottom).
        
        Returns:
            Sorted list of figure dictionaries.
        """
        # Group by page
        grouped = defaultdict(list)
        for fig in self.figures:
            grouped[fig['page']].append(fig)
        
        sorted_figures = []
        for page in sorted(grouped.keys()):
            page_figures = grouped[page]
            
            if len(page_figures) == 1:
                sorted_figures.append(page_figures[0])
                continue
            
            # Check for two-column layout
            left = [f for f in page_figures if f['bbox'][2] < 0.5]
            right = [f for f in page_figures if f['bbox'][0] >= 0.5]
            
            if left and right:
                left.sort(key=lambda x: x['bbox'][1])
                right.sort(key=lambda x: x['bbox'][1])
                sorted_figures.extend(left)
                sorted_figures.extend(right)
            else:
                page_figures.sort(key=lambda x: x['bbox'][1])
                sorted_figures.extend(page_figures)
        
        return sorted_figures
    
    def merge_figure_titles(self) -> List[Tuple[str, int, List[float]]]:
        """
        Merge figure titles that share the same bounding box.
        
        Returns:
            List of merged (title, page, bbox) tuples.
        """
        titles_dict = defaultdict(list)
        for title, page, bbox in self.figure_titles:
            key = (page, tuple(bbox))
            titles_dict[key].append(title)
        
        merged = []
        for (page, bbox), titles in titles_dict.items():
            merged_title = " ".join(titles).strip()
            merged.append((merged_title, page, list(bbox)))
        
        return merged
