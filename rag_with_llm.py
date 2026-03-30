from rag_core import LocalRAG
from typing import List, Dict
import os

class RAGWithLLM:
    """RAG system with multiple LLM backends."""
    
    def __init__(self, llm_type: str = "ollama", ollama_model: str | None = None, load_vectorstore_on_init: bool = True):
        """
        llm_type options:
        - "ollama": Run locally (free, private, need Ollama installed)
        - "openai": GPT models (paid, API key required)
        - "mock": For testing without LLM
        """
        self.rag = LocalRAG()
        self.llm_type = llm_type
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama2")
        self.load_vectorstore_on_init = load_vectorstore_on_init
        
        # Load vector store (optional during init)
        if self.load_vectorstore_on_init:
            if not self.rag.load_vectorstore():
                print("No existing vector store found. Please run rag_core.py first.")
                raise ValueError("Vector store not initialized")
        
        # Initialize LLM
        self.llm = self._init_llm()
    
    def _init_llm(self):
        """Initialize the selected LLM."""
        
        if self.llm_type == "ollama":
            try:
                from langchain_community.llms import Ollama
                # Download: ollama pull llama2 (or mistral, phi, etc.)
                return Ollama(model=self.ollama_model, temperature=0.7)
            except Exception as e:
                print(f"Ollama not available: {e}")
                print("Falling back to mock LLM. To enable Ollama, install it from https://ollama.ai and run: ollama pull llama2 (or set OLLAMA_MODEL).")
                self.llm_type = "mock"
                return MockLLM()
        
        elif self.llm_type == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        elif self.llm_type == "mock":
            return MockLLM()
        
        else:
            raise ValueError(f"Unknown LLM type: {self.llm_type}")
    
    def ask(self, question: str, k: int = 3) -> Dict:
        """Ask a question and get answer with citations."""

        if not self.rag.vectorstore:
            if not self.rag.load_vectorstore():
                raise ValueError("Vector store not initialized. Run rag_core.py or ingest documents first.")
        
        # 1. Retrieve relevant documents
        results = self.rag.search(question, k=k)
        
        # 2. Format context
        context = self.rag.format_context(results)
        
        # 3. Build prompt
        prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context. 
If the answer cannot be found in the context, say "I don't have information about that in my knowledge base."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        # 4. Get LLM response
        if self.llm_type == "openai":
            response = self.llm.invoke(prompt).content
        else:
            response = self.llm.invoke(prompt)
        
        # 5. Extract citations
        citations = []
        for i, (doc, score) in enumerate(results):
            citations.append({
                "source": doc.metadata.get('source', 'unknown'),
                "page": str(doc.metadata.get('page', 'N/A')),
                "relevance": float(1 - score),
                "excerpt": doc.page_content[:200]
            })
        
        return {
            "question": question,
            "answer": response,
            "citations": citations,
            "context_used": context
        }
    
    def chat(self):
        """Interactive chat loop."""
        print("\n" + "="*50)
        print("RAG Chatbot Ready! (type 'quit' to exit)")
        print("="*50)
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            print("\nThinking...")
            result = self.ask(user_input)
            
            print(f"\n🤖 Assistant: {result['answer']}")
            print("\n📚 Sources:")
            for citation in result['citations']:
                print(f"  - {citation['source']} (relevance: {citation['relevance']:.2f})")


class MockLLM:
    """Mock LLM for testing without API calls."""
    
    def invoke(self, prompt: str) -> str:
        # Simple extraction without real LLM
        if "RAG" in prompt or "Retrieval" in prompt:
            return "RAG combines information retrieval with language generation to provide accurate, citeable answers."
        elif "production" in prompt.lower():
            return "Production AI systems require versioning, monitoring, and scaling infrastructure."
        else:
            return "Based on the provided context, I can help answer your question. Please check the citations for specific information."


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    import sys
    
    # Option 1: Use Ollama (if installed)
    print("Starting RAG System with Ollama...")
    rag_system = RAGWithLLM(llm_type="ollama")
    
    # Option 2: Use mock LLM (no setup required)
    # rag_system = RAGWithLLM(llm_type="mock")
    
    # Option 3: Use OpenAI
    # rag_system = RAGWithLLM(llm_type="openai")
    
    # Interactive chat
    rag_system.chat()