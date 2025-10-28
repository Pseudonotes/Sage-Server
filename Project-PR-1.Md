## PR on android app
- Currently app allows document uploads and queries document name as part of its search mechanism.
- This is great until a document is just not  what the name indicates. Also this works well when the files are still on your device.
- The essence is to improve user experience in a way that the documents are analyzed for semantic meanin durin uploads.

### What this PR aims to improve.
- Approach the project in a more ML manner.
- Add semantic meaning to documents not just plain pdf uploads as is currently

### Requirements for the PR to be successful.
1. #### A RAG pipeline working as follows
    - Make use of a suitable <b>embeddin model</b> to split input into vectors for the LLM.
    - <em>Store this embeddings</em> used for retrieval of documents and search.
    - Vectorize user query and mappin it to whats in store and use it to retrieve results.
    - On each input file processed : 
        - A stream of topics from document will be sent to subscribers of such topics.
        - Document vector stored on a vector database.
        - Cache a map of document url to keywords for faster retrieval.

    - On each input query.
        - Check if our cache has the document/topic in question
        - If cache miss vectorize input and forward to process
        - Retrieve all <em><b>nearest-K documents</b></em>
        

2. ### On-device processing
    > <b>Use case</b> \
    > Lets say I have a bunch of pdf files maybe course content that I wish to read off. I would then upload this files that will be processed on my device <i>(offline)</i> and allow me to either Converse with a context aware LLM<i>(Chatbot)</i>, Outline Key topics and summarys <i>(Summarizer)</i>, Search for specific components in the document store and expound <i>(explorer)</i>.

    Am still explorin the options available for on device since the models are sometimes so large.


### Stack in question
- The current frontend will remain as is android app UI.
- Develop a backend system in fast API
- Store vectors in redis and any other cache form.


### Preparations on android app
- Adopt a modular structure.
- Adopt feature modules.
- Abstract code into fundamental modules.

### Milestones
- Backend Service [X]
    - Accepts file upload [X]
    - Accepts dynamic file resolution.<em>(download link)</em>[X]
    - Processes file in background and store embeddings
    - retrieves related embeddings to user query
    - transforms embeddings to attributes
- [API doc]('http://localhost:8000/openapi.json') [X]

