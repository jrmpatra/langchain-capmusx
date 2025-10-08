from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

text = """LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more."""
docs = PyPDFLoader("file.pdf").load()
docs1  = PyPDFLoader("file.pdf").load_and_split()


#splitter = CharacterTextSplitter(separator="", chunk_size=100, chunk_overlap=20)
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

#Document Bassed Splitting
#splitter = RecursiveCharacterTextSplitter.from_language(language="Language.PYTHON", chunk_size=100, chunk_overlap=0)

#Semantic Text Splitter
#splitter = SemanticTextSplitter.from_model_name(model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_size=100, chunk_overlap=0)   

result = splitter.split_documents(docs1)
print(result[-1].page_content)
print(len(result))