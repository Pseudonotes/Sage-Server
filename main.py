from fastapi import FastAPI, UploadFile, BackgroundTasks, WebSocket

app = FastAPI(debug=True, title="Sage")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

from pypdf import PdfReader

import re, redis, requests
import numpy as np

import pika
import datetime

from redis.commands.search.query import Query

import firebase_admin
from firebase_admin import credentials, firestore, messaging, storage

import os

def init_firebase():
    GOOGLE_APPLICATION_CREDENTIALS= os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') is None:
        return
    cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    firebase_admin.initialize_app(credential=cred)
    db=firestore.client()
    bucket = storage.bucket("gs://apptales-6c0f3.appspot.com")


redis_client = redis.Redis(host='localhost', port=6379)

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

FILE_WORKER_TOPIC="FILE_WORKER_TOPIC"

from transformers import pipeline

from pydantic import BaseModel


class Parser():
    """Parser performs all text operations.
    """
    
    def __init__(self):
        pass

    def pdf_to_text(self, path) -> str:
        reader = PdfReader(path)
        return '\n'.join([p.extract_text() for p in reader.pages])

    def clean_text_output(self, text) -> str :
        return re.sub(r'\s+', ' ', text).strip()

class Embedder():
    """initialized by a model. Allows for chunk and embedd operatins
    """
    
    def __init__(self, model):
        self.model = model

    def chunk_text(self, text, max_tokens=500):
        return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]

    def embedd_text(self, chunks):
        return self.model.encode(chunks, show_progress_bar=True)

class Summarizer():
    def __init__(self,model:str = "Falconsai/text_summarization"):
        self.model=model
        self.pipe=pipeline("summarization", model)
    
    def summarize(self,text_input:str, max:int=1000, min:int=20):
        return self.pipe(text_input, max, min)[0]['summary_text']


parser = Parser()
embedder = Embedder(model=model)
summarizer = Summarizer()

channel.queue_declare(queue=FILE_WORKER_TOPIC)
    
def file_topic_callback(ch, method, properties, body):
    print(ch, method, body, properties)
    pass
    # with open('out.txt','a') as file:
    #     file.write(f'{body}\n')
    #     file.close()

# def start_out(token:str|None):
#     """A subscriber on file processin events
    
#     Keyword arguments:
#     argument -- description
#     Return: return_description
#     """
    
#     channel_reciever = connection.channel()
#     channel_reciever.basic_consume(queue=FILE_WORKER_TOPIC, on_message_callback=file_topic_callback, auto_ack=True, arguments=token)
#     channel_reciever.start_consuming()
            

def process_input_file(file, ref:str, tasks : BackgroundTasks):
    #print(f'processin {ref}')
    channel.basic_publish(exchange='',routing_key=FILE_WORKER_TOPIC, body=f'processin {ref}')
    # tasks.add_task(start_out, None)
    # create event listener for file
    text = parser.pdf_to_text(file)
    chunks = embedder.chunk_text(parser.clean_text_output(text))
    channel.basic_publish(exchange='',routing_key=FILE_WORKER_TOPIC, body=f'Total chunks {len(chunks)} of 500 each.')
    # embeddings = embedder.embedd_text(chunks).astype(np.float32).tobytes()
    for i, chunk in enumerate(chunks):
        emb = embedder.embedd_text([chunk]).astype(np.float32).tobytes()
        # add each chunk with metadata
        channel.basic_publish(exchange='',routing_key=FILE_WORKER_TOPIC, body=f'Processin chunk {i}/{len(chunks)}.')
        redis_client.vset().vadd(
            "pdf_vectors",      # vector set
            emb,                # embedding vector
            f"{ref}_chunk{i}",  # unique ID per chunk
            attributes={          # store document info
                "filename": ref,
                "chunk_text": chunk
            }
        )
    # redis_client.vset().vadd(ref, embeddings, ref)
    channel.basic_publish(exchange='',routing_key=FILE_WORKER_TOPIC, body=f'Processin {ref} Done.')
    # print(f'done processin {ref}')
    channel.close()

class UploadOptions(BaseModel):
    notification_token:str=None

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile, task: BackgroundTasks,options:UploadOptions=None):
    print(options)
    channel.basic_publish(exchange='',routing_key=FILE_WORKER_TOPIC, body=f'uploaded {file.filename}')
    task.add_task(process_input_file, file.file, file.filename.split('.')[0].strip(), task)
    return {"filename": file.filename, "status":"process in background."}


def notify_device(token, topic, data):
    messaging.send(message=messaging.Message(data=data, token=token,topic=topic))

@app.post("/process-file/")
async def create_upload_file(fileUrl : str, deviceToken : str|None, task: BackgroundTasks):
    """Processes a file from a download url.
    
    Arguments:
    fileUrl -- endpoint of file resource
    Return: filename + status
    """
    # download file
    print(f'url = {fileUrl}')
    
    local_filename=f'temp_{datetime.datetime.utcnow()}.pdf'

    res = requests.get(fileUrl, stream=True)

    print(res.status_code)

    if res.status_code == 200:

        with open(local_filename, 'wb') as f:
            # bucket.blob(fileUrl).download_to_file_name(f)
            f.write(res.content)

            # notify subscribers
            channel.basic_publish(exchange='',routing_key=FILE_WORKER_TOPIC, body=f'Downloaded {file.filename}')
            task.add_task(process_input_file, f, local_filename, deviceToken, task)
            return {"success": True, "status":"Processing {local_filename} in background."}
    else:
        return {"success": False, "status":"Failed to download file."}
    


@app.get('/search')
async def search(query:str, limit:int=2):

    cache_query_result = redis_client.hgetall(f'query:{query}')
    if len(cache_query_result) != 0 :
        if 'files' in cache_query_result.keys():
            cache_query_result['files']=cache_query_result['files'].split('#')
        return cache_query_result
    else:
        # embedd query
        query_chunks = embedder.chunk_text(parser.clean_text_output(query), max_tokens=20)
        embed = embedder.embedd_text(query_chunks).astype(np.float32).tobytes()


        # Run similarity search with metadata returned
        results = redis_client.vset().vsim(
            "pdf_vectors",
            embed,
            count=limit,
            # with_scores=True,
        )

        # Extract vector IDs
        # vector_ids = [r.split('_') for i,r in enumerate(results)]

        # Retrieve all attributes for these vectors
        attrs = [
                redis_client.vset().vgetattr(
                "pdf_vectors",
                results[i],
                )
                for i in range(len(results))
            ]

        # file_names = [attr[list(attr.keys())[0]] for attr in attrs]
        chunks = [attr['chunk_text'] for attr in attrs]
        chunk_text='\n'.join(chunks)
        # print('##########################################')['chunk_text']
        # print(chunk_text)
        # print('##########################################')
        res= summarizer.summarize(
            f"""
            User Query : {query}
            Context:
            {chunk_text}
            """
            )

        # create a pub/sub structure for current query
        # do in back notify all subs or emit
        # chunks_summary=[summarizer.summarize(c[:300]) for c in chunks]
        files=[a['filename'] if 'filename' in a.keys() else '#' for a in attrs]

        ret= {
            # "results": results, 
            # "mapped" : attrs,
            "res":res,
            "metadata":{
                # "file_names":set(file_names),
                "chunks":len(attrs),
                "files":set(files)
                # "chunk_text":chunks_summary,
                # "attributes":attrs
            }
        }
        # print(files)
        redis_client.hmset(f'query:{query}', {"res":res, "files":'#'.join(set(files))})
        return ret