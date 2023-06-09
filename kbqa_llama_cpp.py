import os

from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Qdrant
from lib.yuque_loader import YuqueLoader


def load_docs(**kwargs):
    loader = YuqueLoader(
        url=os.environ["YUQUE_BASE_URL"],
        token=os.environ["YUQUE_API_TOKEN"],
        user_agent="kbqa",
    )
    documents = loader.load(**kwargs)
    text_splitter = TokenTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(documents)
    return docs


def embed_docs(docs):
    embeddings = LlamaCppEmbeddings(model_path=os.environ["LLAMA_CPP_MODEL_PATH"])
    Qdrant.from_documents(
        docs, embeddings,
        host=os.environ["QDRANT_HOST"], prefer_grpc=True,
        collection_name="yuque",
    )


def create_qa():
    import qdrant_client
    from langchain.llms import LlamaCpp
    embeddings = LlamaCppEmbeddings(model_path=os.environ["LLAMA_CPP_MODEL_PATH"])
    qdrant = Qdrant(client=qdrant_client.QdrantClient(host=os.environ["QDRANT_HOST"], prefer_grpc=True),
                    collection_name="yuque",
                    embeddings=embeddings)
    retriever = qdrant.as_retriever()
    llm = LlamaCpp(model_path=os.environ["LLAMA_CPP_MODEL_PATH"], n_ctx=2048)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa


if __name__ == "__main__":
    qa = create_qa()
    # read question from stdin and print answer repeatedly
    while True:
        question = input("Question: ")
        answer = qa.run(question)
        print(answer)
