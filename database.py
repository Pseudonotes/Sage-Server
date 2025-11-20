# from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from  sentence_transformers import SentenceTransformer
import numpy as np
import redis
from summarizer import Summarizer

redis_client = redis.Redis(port=7777)


class VDb():
    def __init__(self):
        embedding_model = SentenceTransformer("google/embeddinggemma-300m")
        self.model = embedding_model

    def load_document_embeddings(self, metadata, chunks):
        print(f'Processin {len(chunks)} {chunks} chunks.')
        docs = [
            (
                chunks[i],
                {
                    **metadata,
                    "chunk_id": f'chunk#{i}'
                },
            )
            for i in range(len(chunks))
        ]

        docs, metadata = [d[0] for d in docs], [d[1] for d in docs]
        embeddings = [
            self.model.encode(d).astype(np.float32).tobytes()
            for d in docs 
        ]
        # print(f'{len(embeddings)} {len(metadata)}: {embeddings[0]}')
        return embeddings, metadata

    

    def store_embeddings_in_redis(self, embeddings,metadata):
        print(f'meta {metadata[i]}')
        for i in range(embeddings):
            redis_client.vset().add(
                "pdf_vectors_store",      # vector set
                embeddings[i],                # embedding vector
                metadata[i]['chunk_id'],  # unique ID per chunk
                attributes={          # store document info
                    **metadata[i]
                }
            )

    def similarity(self, embed, limit=5):
        summarizer = Summarizer()
         # Run similarity search with metadata returned
        print(f'emb : {type(embed)} {embed}')
        results = redis_client.vset().vsim(
            "pdf_vectors_store",
            embed,
            count=limit,
            # with_scores=True,
        )

        # Extract vector IDs
        # vector_ids = [r.split('_') for i,r in enumerate(results)]

        # Retrieve all attributes for these vectors
        attrs = [
                redis_client.vset().vgetattr(
                "pdf_vectors_store",
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
        res= summarizer.pipe().summarize(
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

        return  {res, set(files)}

    






