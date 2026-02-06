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

    @staticmethod
    def _norm_field_name(name: Any) -> str:
        return re.sub(r"[_\s]+", "", str(name or "").strip().lower())

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        if not text:
            return []
        t = str(text).replace(",", "")
        out: List[float] = []
        for m in re.findall(r"-?\d+(?:\.\d+)?", t):
            try:
                out.append(float(m))
            except Exception:
                continue
        return out

    @staticmethod
    def _bucket_age(value: Any) -> Any:
        """
        Bucket ages into:
        18–29, 30–44, 45–59, 60+
        """
        try:
            # numeric-ish
            if isinstance(value, (int, float)):
                age = float(value)
            else:
                s = str(value).strip()
                if not s:
                    return value
                # Already bucketed?
                s_norm = s.replace("—", "–").replace("-", "–").replace(" ", "")
                if s_norm in {"18–29", "30–44", "45–59", "60+"}:
                    return s_norm
                nums = PizzaDataLoader._extract_numbers(s)
                if not nums:
                    return value
                age = (min(nums) + max(nums)) / 2.0 if len(nums) >= 2 else float(nums[0])
        except Exception:
            return value

        # Only bucket plausible adult ages; otherwise leave as-is.
        if age < 18:
            return value
        if age < 30:
            return "18–29"
        if age < 45:
            return "30–44"
        if age < 60:
            return "45–59"
        return "60+"

    @staticmethod
    def _bucket_income(value: Any) -> Any:
        """
        Bucket income into:
        Under $25k, $25k–$49k, $50k–$74k, $75k+
        """
        s = str(value).strip()
        if not s:
            return value

        # Already bucketed? (normalize dashes/spaces)
        s_norm = s.replace("—", "–").replace("-", "–").replace(" ", "")
        canonical = {
            "under$25k": "Under $25k",
            "$25k–$49k": "$25k–$49k",
            "$50k–$74k": "$50k–$74k",
            "$75k+": "$75k+",
        }
        key = s_norm.lower()
        if key in canonical:
            return canonical[key]

        # Parse numeric-ish income values / ranges.
        try:
            if isinstance(value, (int, float)):
                rep = float(value)
            else:
                sl = s.lower().replace(",", "")
                amounts: List[float] = []

                # 25k style
                for m in re.findall(r"(\d+(?:\.\d+)?)\s*k", sl):
                    try:
                        amounts.append(float(m) * 1000.0)
                    except Exception:
                        continue

                # plain numbers (may be 25,000 or 25)
                for m in re.findall(r"-?\d+(?:\.\d+)?", sl):
                    try:
                        amounts.append(float(m))
                    except Exception:
                        continue

                if not amounts:
                    return value

                # Heuristic: if it looks like "25–49" (thousands), upscale.
                if max(amounts) < 1000 and any(tok in sl for tok in ["$", "k", "under", "+", "-", "–", "to"]):
                    amounts = [a * 1000.0 for a in amounts]

                lo = min(amounts)
                hi = max(amounts)
                if "under" in sl:
                    rep = lo - 1.0
                elif "+" in sl or "over" in sl or "more" in sl:
                    rep = hi
                elif len(amounts) >= 2:
                    rep = (lo + hi) / 2.0
                else:
                    rep = float(amounts[0])
        except Exception:
            return value

        if rep < 25000:
            return "Under $25k"
        if rep < 50000:
            return "$25k–$49k"
        if rep < 75000:
            return "$50k–$74k"
        return "$75k+"

    def _normalize_metadata_value(self, field_name: Any, value: Any) -> Any:
        n = self._norm_field_name(field_name)
        if n == "age" or n.endswith("age"):
            return self._bucket_age(value)
        if "income" in n:
            return self._bucket_income(value)
        return value
        
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
                    metadata[col] = self._normalize_metadata_value(col, value)
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
