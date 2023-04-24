from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
import requests
from typing import List, Optional, Dict, Union
from bs4 import BeautifulSoup


class YuqueLoader(BaseLoader):
    """Loader that make request to yuque api and retrieve markdown documents."""
    def __init__(self, url: str, token: str, user_agent: str):
        self.url = url
        self.token = token
        self.user_agent = user_agent

    def load(
            self,
            repo_ids: Optional[List[str]] = None,
            doc_ids: Optional[List[str]] = None,
    ) -> List[Document]:
        """Load data into document objects."""
        headers = {
            "X-Auth-Token": self.token,
            "Content-Type": "application/json",
            "User-Agent": self.user_agent
        }
        if doc_ids is None:
            doc_ids = []
            if repo_ids is None:
                repo_ids = []
                user_resp = requests.get(f"{self.url}/api/v2/user", headers=headers)
                user = user_resp.json()["data"]
                assert user["type"] == "Group", "User is not a group"
                group_id = user["id"]
                response = requests.get(f"{self.url}/api/v2/groups/{group_id}/repos?type=Book", headers=headers)
                repos = response.json()["data"]
                print(f"Loading from {len(repos)} repos...")
                for repo in repos:
                    repo_ids.append(repo["namespace"])

            for repo_id in repo_ids:
                response = requests.get(f"{self.url}/api/v2/repos/{repo_id}/docs", headers=headers)
                docs = response.json()["data"]
                print(f"Loading from {len(docs)} docs in {repo_id}...")
                for doc in docs:
                    doc_ids.append((repo_id, doc['slug']))

        documents = []
        for repo_id, doc_id in doc_ids:
            doc_response = requests.get(f"{self.url}/api/v2/repos/{repo_id}/docs/{doc_id}", headers=headers)
            doc = doc_response.json()["data"]
            if doc["format"] == "lake":
                print(f"Loading lake doc {doc['title']}")
                soup = BeautifulSoup(doc["body_html"], features="html5lib")
                content = soup.get_text()

                metadata: Dict[str, Union[str, None]] = {
                    "title": doc["title"],
                    "category": doc["book"]["name"],
                    "source": f"{self.url}/{repo_id}/{doc_id}",
                }

                documents.append(Document(page_content=content, metadata=metadata))

        return documents
