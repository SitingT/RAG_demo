from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import os
import openai
import sys
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import shutil
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

sys.path.append('../..')

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("/Users/sitingtang/Desktop/Final_Exam_CS5500.pdf"),
    PyPDFLoader("/Users/sitingtang/Desktop/lease.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

splits = text_splitter.split_documents(docs)
embedding = OpenAIEmbeddings()


def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


persist_directory = 'docs/chroma/'

clear_directory(persist_directory)

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
# print(vectordb._collection.count())
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question, k=3)

# print(docs[0].page_content[:100])
# print(docs[0].page_content[:100])

docs_mmr = vectordb.max_marginal_relevance_search(question, k=3)
# print(docs_mmr[0].page_content[:100])
# print(docs_mmr[1].page_content[:100])
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=template,)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    # chain_type="map_reduce"
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
result = qa_chain({"query": question})
print(result["result"])
