"""Document parser - extracts text from PDF, DOCX, TXT, and chunks it."""

import io
from pathlib import Path
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    from PyPDF2 import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain text file (tries multiple encodings)."""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return file_bytes.decode(encoding)
        except (UnicodeDecodeError, AttributeError):
            continue
    return file_bytes.decode("utf-8", errors="replace")


def extract_text_from_csv(file_bytes: bytes) -> tuple[str, dict]:
    """Convert CSV to a natural language description + store schema for Text-to-Pandas."""
    import pandas as pd
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        return "", {}

    schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": list(df.shape),
        "sample_rows": df.head(3).fillna("").astype(str).to_dict(orient="records"),
    }

    description = f"Dataset with {df.shape[0]} rows and {df.shape[1]} columns.\n"
    description += f"Columns: {', '.join(df.columns)}\n\n"
    description += "Column descriptions:\n"
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            description += f"- {col} (numeric): min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"
        else:
            n_unique = df[col].nunique()
            top_values = df[col].value_counts().head(5).index.tolist()
            description += f"- {col} (text/categorical): {n_unique} unique values. Examples: {top_values[:3]}\n"

    # For large datasets, keep description concise (Text-to-Pandas will query the full file)
    if df.shape[0] > 100000:
        description += f"\nNote: Large dataset ({df.shape[0]:,} rows). Use data queries for detailed analysis.\n"

    return description, schema


def parse_file(filename: str, file_bytes: bytes) -> tuple[str, str, dict]:
    """
    Parse a file and return (text, file_type, metadata).
    For CSVs, metadata includes the schema for Text-to-Pandas.
    """
    ext = Path(filename).suffix.lower()
    metadata = {}

    if ext == ".pdf":
        text = extract_text_from_pdf(file_bytes)
        file_type = "pdf"
    elif ext == ".docx":
        text = extract_text_from_docx(file_bytes)
        file_type = "docx"
    elif ext in (".txt", ".md", ".log"):
        text = extract_text_from_txt(file_bytes)
        file_type = "txt"
    elif ext in (".csv", ".tsv"):
        text, metadata = extract_text_from_csv(file_bytes)
        file_type = "csv"
        metadata["queryable"] = True
    elif ext in (".xlsx", ".xls"):
        import pandas as pd
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        best_df = None
        for sheet in xl.sheet_names:
            sheet_df = pd.read_excel(xl, sheet_name=sheet)
            if best_df is None or len(sheet_df) > len(best_df):
                best_df = sheet_df
        df = best_df if best_df is not None else pd.DataFrame()
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        text, metadata = extract_text_from_csv(buf.getvalue())
        file_type = "excel"
        metadata["queryable"] = True
    else:
        text = extract_text_from_txt(file_bytes)
        file_type = "unknown"

    return text, file_type, metadata


def chunk_text(text: str) -> list[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    if not text or not text.strip():
        return []
    chunks = _splitter.split_text(text)
    return [c for c in chunks if c.strip()]
