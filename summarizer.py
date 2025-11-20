
class Summarizer():
    def __init__(self):
        self.model=None

    def pipe(self,model:str = "Falconsai/text_summarization"):
        self.model=model
        self.pipe = pipeline("summarization", model)
        return self

    def summarize(self,text_input:str, max:int=1000, min:int=20):
        return self.pipe(text_input, max, min)[0]['summary_text']
