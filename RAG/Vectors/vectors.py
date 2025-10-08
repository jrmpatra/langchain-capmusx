from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Create Document objects
loader = PyPDFLoader('file.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splitted_docs = splitter.split_documents(docs)

# vector_store = Chroma(
#     embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
#     persist_directory='my_chroma_db',
#     collection_name='sample'
# )

vector_store = FAISS.from_documents(
                splitted_docs, 
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )


# add documents
vector_store.add_documents(docs)

results = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)

#print(results)
# vector_store.similarity_search_with_score(
#     query="",
#     filter={"team": "Chennai Super Kings"}
# )

