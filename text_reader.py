from pdfminer.high_level import extract_text
from llama_index.core.node_parser import SentenceSplitter
from typing import List


class PDFProcessor:
    def __init__(self, chunk_size=1024, chunk_overlap=200):
        """Initialize the PDFProcessor with SentenceSplitter."""
        self._splitter = SentenceSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )


    def _extract_text_from_document(self, pdf_file):
        """Extract text from a PDF file."""
        try:
            text = extract_text(pdf_file)
            return text
        except FileNotFoundError:
            print(f"File not found: {pdf_file}")
        except PermissionError:
            print(f"Permission denied for file: {pdf_file}")
        except Exception as e:
            print(f"An error occurred while extracting text from the PDF: {e}")
        return None


    def process_document(self, pdf_file) -> List[str]:
        """Process the PDF file and split the extracted text."""
        text = self._extract_text_from_document(pdf_file)
        if text is None:
            print("No text extracted. Processing halted.")
            return []
        
        try:
            nodes = self._splitter.split_texts([text])
            print("Number of split raw text pieces:", len(nodes))
            return nodes
        except Exception as e:
            print(f"An error occurred during text splitting: {e}")
            return []
