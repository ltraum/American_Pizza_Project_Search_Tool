"""
Data loader for pizza interview and survey data
Handles Excel file reading and preprocessing
"""
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

from config import EXCEL_FILE

# Columns that contain narrative/longform interview text (indexed as searchable content).
# All other columns are metadata-only and must not be concatenated into document text.
NARRATIVE_COLUMN_PATTERN = re.compile(r"^q[1-5]_response$", re.IGNORECASE)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PizzaDataLoader:
    """Load and preprocess pizza interview/survey data"""
    
    def __init__(self, excel_file: Path = None):
        self.excel_file = excel_file or EXCEL_FILE
        self.data = None
        self.processed_data = []
        
    def load_data(self) -> pd.DataFrame:
        """Load data from Excel file"""
        if not self.excel_file.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_file}")
        
        logger.info(f"Loading data from {self.excel_file}")
        self.data = pd.read_excel(self.excel_file)
        logger.info(f"Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
        return self.data
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data"""
        if self.data is None:
            self.load_data()
        
        return {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "null_counts": self.data.isnull().sum().to_dict(),
            "sample": self.data.head(3).to_dict('records')
        }
    
    def preprocess_for_search(self) -> List[Dict[str, Any]]:
        """
        Preprocess data for search indexing.
        Searchable document text is built only from narrative columns (q1_response..q5_response).
        All columns are stored in metadata for filtering/display; metadata is not concatenated
        into the text so results show narrative excerpts only.
        """
        if self.data is None:
            self.load_data()
        
        logger.info("Preprocessing data for search...")
        processed = []
        
        for idx, row in self.data.iterrows():
            # Build searchable text only from narrative/longform response columns
            text_parts = []
            metadata = {}
            
            for col in self.data.columns:
                value = row[col]
                if pd.notna(value):
                    metadata[col] = value
                    # Only narrative columns go into document text (no metadata prefix in excerpts)
                    if NARRATIVE_COLUMN_PATTERN.match(col):
                        text_parts.append(str(value))
            
            doc = {
                "id": str(idx),
                "text": " ".join(text_parts) if text_parts else "",
                "metadata": metadata,
                "row_index": idx
            }
            processed.append(doc)
        
        self.processed_data = processed
        logger.info(f"Preprocessed {len(processed)} documents")
        return processed
    
    def get_text_columns(self) -> List[str]:
        """Identify text columns in the dataset"""
        if self.data is None:
            self.load_data()
        
        text_cols = []
        for col in self.data.columns:
            dtype = str(self.data[col].dtype)
            if dtype in ['object', 'string']:
                text_cols.append(col)
        
        return text_cols
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all preprocessed documents"""
        if not self.processed_data:
            self.preprocess_for_search()
        return self.processed_data
