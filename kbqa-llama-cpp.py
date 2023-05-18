import os
import sys

from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Qdrant
from lib.yuque_loader import YuqueLoader


__globals = {}


def load_docs(**kwargs):
    loader = YuqueLoader(
        url=os.environ["YUQUE_BASE_URL"],
        token=os.environ["YUQUE_API_TOKEN"],
        user_agent="kbqa",
    )
    documents = loader.load(**kwargs)
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def embed_docs(docs):
    embeddings = LlamaCppEmbeddings(model_path=__globals["model_path"])
    Qdrant.from_documents(
        docs, embeddings,
        url="localhost:6334", prefer_grpc=True,
        collection_name="maxtropy_yuque",
    )


def create_qa():
    import qdrant_client
    from langchain.llms import LlamaCpp
    embeddings = LlamaCppEmbeddings(model_path=__globals["model_path"])
    qdrant = Qdrant(client=qdrant_client.QdrantClient(url="localhost:6334", prefer_grpc=True),
                    collection_name="maxtropy_yuque",
                    embeddings=embeddings)
    retriever = qdrant.as_retriever()
    llm = LlamaCpp(model_path=__globals["model_path"])
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa


if __name__ == "__main__":
    # get model path from 1st argument
    __globals["model_path"] = sys.argv[1]
    qa = create_qa()
    # read question from stdin and print answer repeatedly
    while True:
        question = input("Question: ")
        answer = qa.run(question)
        print(answer)
