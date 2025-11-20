import io
import asyncio
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from database import VDb 
from parser import Parser

app = FastAPI(
    title="Simple RAG Document Processor",
    description="Processes and queries documents using in-memory vector storage."
)

parser = Parser()
database = VDb()

async def prep_pdf_task(file):
    sleep()
    file_content, filename = file.file, file.filename
    try:
        print(f"Starting processing for file: {filename}")
        
        paragraphs, metadata = parser.extract_paragraphs(file_content)
        
        chunks = parser.chunk_pdf(paragraphs)
        
        print(f'Total paragraphs extracted: {len(paragraphs)}')
        print(f'Total chunks created: {len(chunks)}')
        
        database.load_document_embeddings(
            metadata={**metadata, "filename":filename},
            chunks=chunks
        )
        print(f"Finished processing and embedding for file: {filename}")

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

@app.post('/upload', summary="Upload a PDF for processing")
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    # file_content = await file.read()
    background_tasks.add_task(prep_pdf_task, file)
    
    return {
        "success": True,
        "data": f"PDF file '{file.filename}' received and processing started in the background."
    }


@app.get('/search', summary="Query the uploaded documents (RAG retrieval step)")
async def search(q:str, limit:int=5):
    if not q:
        raise HTTPException(status_code=400, detail="Query string 'q' cannot be empty.")
    
    query_chunks = parser.chunk_pdf(parser.clean_text_output(q), 20)
    embeddings,_ = database.load_document_embeddings(chunks=query_chunks,metadata={"query":q,})
    print(f'embeddings : {embeddings} {type(embeddings)}')
    res = database.similarity(embed=embeddings,limit=limit)

    if not res:
        return {
            "success": True,
            "data": "No relevant documents found in the vector store."
        }
        

    print(f'Res : {res}')
    formatted_results = [
        {"source": result[0].metadata.get('source', 'Unknown'), "text": result[0].page_content, "score": result[1]}
        for result in res
    ]
    # print(files)
    # redis_client.hmset(f'query:{query}', {"res":res, "files":'#'.join(set(files))})
    return {
            "success": True,
            "data": {
                "query": q,
                "retrieved_chunks": formatted_results
            }
    }