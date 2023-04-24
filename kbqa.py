import os
import pinecone
from langchain import OpenAI
from langchain.chains import RetrievalQA

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Pinecone
from lib.yuque_loader import YuqueLoader


def load_docs():
    loader = YuqueLoader(
        url=os.environ["YUQUE_BASE_URL"],
        token=os.environ["YUQUE_API_TOKEN"],
        user_agent="kbqa",
    )
    documents = loader.load(repo_ids=["hqtk2d/xzfg1a"])
    text_splitter = TokenTextSplitter(chunk_size=6000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def embed_docs(docs):
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    index_name = "openai"
    Pinecone.from_documents(docs, embeddings, index_name=index_name)


def create_qa():
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    index_name = "openai"
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    retriever = docsearch.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    return qa


if __name__ == "__main__":
    qa = create_qa()
    # read question from stdin and print answer repeatedly
    while True:
        question = input("Question: ")
        answer = qa.run(question)
        print(answer)