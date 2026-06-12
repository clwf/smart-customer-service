"""
knowledge_base.py
RAG 知识库：支持持久化，只在首次或文档更新时构建
"""

import os
import hashlib
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from modelscope import snapshot_download


class KnowledgeBase:
    """公司知识库，支持文档检索，向量数据库持久化"""

    def __init__(self, docs_dir: str = "./documents", db_dir: str = "./chroma_db"):
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.vectorstore = None
        self.retriever = None
        self._init()

    def _get_docs_hash(self) -> str:
        """计算文档目录的哈希值，用来判断文档有没有变化"""
        hash_obj = hashlib.md5()
        for root, dirs, files in os.walk(self.docs_dir):
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                # 用文件名 + 修改时间 + 文件大小来计算哈希
                stat = os.stat(filepath)
                hash_obj.update(f"{filename}{stat.st_mtime}{stat.st_size}".encode())
        return hash_obj.hexdigest()

    def _load_embedding_model(self):
        """加载 Embedding 模型"""
        model_dir = os.path.join("models", "AI-ModelScope", "bge-small-zh-v1___5")

        if not os.path.exists(model_dir):
            print("  首次下载 Embedding 模型...")
            model_dir = snapshot_download(
                "AI-ModelScope/bge-small-zh-v1.5", cache_dir="./models"
            )

        return HuggingFaceEmbeddings(
            model_name=model_dir,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_documents(self):
        """加载并切分文档"""
        loader = DirectoryLoader(
            self.docs_dir,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " "],
        )
        return splitter.split_documents(docs)

    def _init(self):
        """初始化：判断是否需要重新构建"""
        embedding = self._load_embedding_model()

        # 检查是否已经有持久化的向量数据库
        hash_file = os.path.join(self.db_dir, ".docs_hash")
        current_hash = self._get_docs_hash()

        db_exists = os.path.exists(self.db_dir) and os.path.isdir(self.db_dir)
        hash_match = False

        if db_exists and os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                saved_hash = f.read().strip()
            hash_match = saved_hash == current_hash

        if hash_match and db_exists:
            # 文档没有变化，直接加载已有的向量数据库
            print("正在加载已有的知识库（跳过构建）...")
            self.vectorstore = Chroma(
                persist_directory=self.db_dir,
                embedding_function=embedding,
            )
            print("知识库加载完成！")
        else:
            # 首次运行 或 文档有变化，重新构建
            print("正在构建知识库...")
            chunks = self._load_documents()
            print(f"  切分成 {len(chunks)} 个文本块")

            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                persist_directory=self.db_dir,
            )

            # 保存哈希值
            os.makedirs(self.db_dir, exist_ok=True)
            with open(hash_file, "w") as f:
                f.write(current_hash)

            print("  知识库构建完成并已保存到磁盘！")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def search(self, query: str) -> str:
        """检索知识库"""
        docs = self.retriever.invoke(query)
        if not docs:
            return "未找到相关信息。"

        results = []
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            filename = source.split("\\")[-1].split("/")[-1]
            results.append(f"[来源: {filename}]\n{doc.page_content}")

        return "\n\n---\n\n".join(results)
