from langchain_huggingface import HuggingFaceEmbeddings

emebeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = ["This is a sample text to be embedded."]
output = emebeddings.embed_documents(text)
print(output)