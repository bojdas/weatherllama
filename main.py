# use ollama and langchain to process weather data
# before running this, run 'ollama serve' on local machine


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


loader = CSVLoader(file_path="boston_2023_temps.csv")

data = loader.load()

# selected_model="llama2"
selected_model="mistral"

# Create embeddings
embeddings = OllamaEmbeddings(model=selected_model)

# a db to store the data for retrieval
chroma_db = Chroma.from_documents(
    data, embeddings
)

llm = Ollama(model=selected_model)


prompt_template = PromptTemplate(
    input_variables=["context"],
    template="Given this context: {context}, please directly answer the question: {question}.",
)

# Set up the question-answering chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=chroma_db.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template},
)


# setup for retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=chroma_db.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template},
)
print(chroma_db.as_retriever())
result = qa_chain({"query": "what is the highest temperature in Jan for Boston in the loaded dataset"})
print(f" llama2 responded with {result}")
print(result)

