class Embedder():
    """initialized by a model. Allows for chunk and embedd operatins
    """
    
    def __init__(self, model):
        self.model = model

    def chunk_text(self, text, max_tokens=500):
        return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]

    def embedd_text(self, chunks):
        return self.model.encode(chunks, show_progress_bar=True)
