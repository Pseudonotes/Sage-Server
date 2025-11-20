import re
from pypdf import PdfReader

class Parser():
    """Parser performs all text operations.
    """
    
    def __init__(self):
        pass

    def extract_paragraphs(self,file):
        text,metadata = self.pdf_to_text(file)
        all_text = self.clean_text_output(text)
        paragraphs = [p.strip() for p in all_text.split("\n\n") if p.strip()] 
        return paragraphs, metadata

    def pdf_to_text(self, file):
        reader = PdfReader(file)
        return '\n'.join([p.extract_text()+"\n\n" for p in reader.pages]), reader.metadata

    def clean_text_output(self, text) -> str :
        return re.sub(r'\s+', ' ', text).strip()

    def chunk_pdf(self,paragraphs, chunk_size=1000) -> [str]:
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            estimated_new_length = len(current_chunk) + len(paragraph) + 2 

            if estimated_new_length <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(paragraph) > chunk_size:
                    chunks.append(paragraph)
                    current_chunk = "" # Reset current_chunk
                else:
                    current_chunk = paragraph

            # Don't forget the last chunk!
            if current_chunk:
                chunks.append(current_chunk)

        return chunks
