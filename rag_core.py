import os
from typing import List, Tuple
from pathlib import Path

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class LocalRAG:
    """RAG system using local models - no API keys needed."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        embeddings_backend = os.getenv("EMBEDDINGS_BACKEND", "hf")
        if embeddings_backend == "fake":
            self.embeddings = FakeEmbeddings(size=384)
            print("✓ Using fake embeddings (hosted demo)")
        else:
            try:
                # Using sentence-transformers (runs locally, free, fast)
                print("Loading embedding model (first time downloads ~400MB)...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                print("✓ Embedding model loaded")
            except Exception as e:
                print(f"Embedding model unavailable: {e}")
                print("Falling back to fake embeddings. Set EMBEDDINGS_BACKEND=fake to silence this.")
                self.embeddings = FakeEmbeddings(size=384)
        
        # Initialize vector store
        self.vectorstore = None
        
    def load_documents(self, directory: str) -> List[Document]:
        """Load all PDF and TXT files from directory."""
        documents = []
        path = Path(directory)
        
        for file_path in path.glob("*"):
            documents.extend(self._load_file(file_path))
        
        print(f"\nTotal documents loaded: {len(documents)}")
        return documents

    def _load_file(self, file_path: Path) -> List[Document]:
        if file_path.suffix.lower() == '.pdf':
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                print(f"  Loaded PDF: {file_path.name} ({len(docs)} pages)")
                return docs
            except Exception as e:
                print(f"  Failed to load {file_path.name}: {e}")
                return []
        
        if file_path.suffix.lower() == '.txt':
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                print(f"  Loaded TXT: {file_path.name}")
                return docs
            except Exception as e:
                print(f"  Failed to load {file_path.name}: {e}")
                return []

        return []
    
    def chunk_documents(self, documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Add metadata for citation
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = 'unknown'
        
        return chunks
    
    def create_vectorstore(self, chunks: List[Document]):
        """Create or update vector database."""
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        print(f"✓ Vector store created with {len(chunks)} chunks")

    def add_documents(self, file_paths: List[str], chunk_size: int = 500, chunk_overlap: int = 50) -> int:
        """Add new documents to the existing vector store."""
        documents: List[Document] = []
        for file_path in file_paths:
            documents.extend(self._load_file(Path(file_path)))

        if not documents:
            return 0

        chunks = self.chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not self.vectorstore:
            self.create_vectorstore(chunks)
        else:
            self.vectorstore.add_documents(chunks)
            self.vectorstore.persist()

        return len(chunks)
    
    def load_vectorstore(self):
        """Load existing vector database."""
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"✓ Loaded existing vector store")
            return True
        return False
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for relevant chunks."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Load or create first.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def format_context(self, results: List[Tuple[Document, float]], max_chars: int = 2000) -> str:
        """Format search results as context for LLM."""
        context_parts = []
        
        for i, (doc, score) in enumerate(results):
            # Extract source info
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'N/A')
            
            # Truncate content if needed
            content = doc.page_content[:max_chars // len(results)]
            
            context_parts.append(f"""
[Source {i+1}] (Document: {source}, Page: {page}, Relevance: {1-score:.2f})
{content}
""")
        
        return "\n".join(context_parts)

# ============================================
# Test with Sample Documents
# ============================================
def create_sample_documents():
    """Create sample text files for testing."""
    os.makedirs("documents", exist_ok=True)
    
    samples = [
        ("ai_intro.txt", """
Artificial Intelligence (AI) is the simulation of human intelligence in machines. 
Key subfields include machine learning, deep learning, natural language processing, 
and computer vision. Modern AI systems use neural networks with millions of parameters.
"""),
        
        ("rag_explained.txt", """
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval 
with language generation. RAG systems first search a knowledge base for relevant documents, 
then use those documents as context for an LLM. This reduces hallucinations and allows 
the model to cite its sources.
"""),
        
        ("python_tips.txt", """
Python is great for AI development. Key libraries include NumPy for numerical computing,
PyTorch and TensorFlow for deep learning, and LangChain for building RAG applications.
Virtual environments help manage dependencies across projects.
"""),
        
        ("production_ai.txt", """
Deploying AI to production requires: model versioning, A/B testing, monitoring drift,
and scaling infrastructure. Tools like Docker, Kubernetes, and MLflow are essential.
Always include logging and alerting for production systems.
""")
    ]
    
    for filename, content in samples:
        with open(f"documents/{filename}", "w") as f:
            f.write(content)
    
    print("Created sample documents in ./documents/")

if __name__ == "__main__":
    # Test the RAG system
    print("=" * 50)
    print("Testing Local RAG System")
    print("=" * 50)
    
    # Create test documents
    create_sample_documents()
    
    # Initialize RAG
    rag = LocalRAG()
    
    # Load and process documents
    docs = rag.load_documents("documents")
    chunks = rag.chunk_documents(docs)
    rag.create_vectorstore(chunks)
    
    # Test search
    test_queries = [
        "What is RAG?",
        "How do I deploy AI to production?",
        "What Python libraries are used for AI?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*40}")
        print(f"Query: {query}")
        print('='*40)
        
        results = rag.search(query, k=2)
        context = rag.format_context(results)
        print(f"Retrieved context:\n{context[:500]}...")