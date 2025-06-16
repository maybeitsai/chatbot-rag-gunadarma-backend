import os
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from rag.vector_store import VectorStoreManager
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self):
        self.llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash-preview-05-20")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            google_api_key=self.google_api_key,
            temperature=0
        )
        
        # Initialize vector store
        self.vector_store_manager = VectorStoreManager()
        self.vector_store = self.vector_store_manager.get_vector_store()
        
        # Setup retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Anda adalah asisten AI yang hanya menjawab berdasarkan informasi yang tersedia dalam dokumen sumber yang diberikan.

ATURAN PENTING:
1. Hanya gunakan informasi dari konteks yang diberikan di bawah ini
2. Jika informasi tidak tersedia dalam konteks, jawab dengan: "Maaf, informasi tersebut tidak tersedia dalam data kami."
3. Selalu sertakan URL sumber dalam jawaban Anda
4. Jangan menambahkan informasi dari pengetahuan umum atau spekulasi
5. Jawab dalam bahasa Indonesia yang jelas dan informatif

Konteks dari dokumen:
{context}

Pertanyaan: {question}

Jawaban:
"""
        )
        
        # Setup QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Process question and return answer with sources"""
        try:
            # Get answer from QA chain
            result = self.qa_chain({"query": question})
            
            answer = result["result"]
            source_documents = result["source_documents"]
            
            # Extract source URLs
            source_urls = []
            for doc in source_documents:
                url = doc.metadata.get("url", "")
                if url and url not in source_urls:
                    source_urls.append(url)
            
            # Check if answer indicates no information found
            no_info_phrases = [
                "maaf, informasi tersebut tidak tersedia",
                "tidak tersedia dalam data kami",
                "tidak ada informasi",
                "tidak ditemukan informasi"
            ]
            
            status = "not_found" if any(phrase in answer.lower() for phrase in no_info_phrases) else "success"
            
            return {
                "answer": answer,
                "source_urls": source_urls,
                "status": status,
                "source_count": len(source_documents)
            }
            
        except Exception as e:
            return {
                "answer": f"Maaf, terjadi kesalahan dalam memproses pertanyaan: {str(e)}",
                "source_urls": [],
                "status": "error",
                "source_count": 0
            }
    
    def test_connection(self) -> bool:
        """Test if the RAG pipeline is working"""
        try:
            test_result = self.ask_question("Apa itu Universitas Gunadarma?")
            return test_result["status"] in ["success", "not_found"]
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

if __name__ == "__main__":
    # Test the RAG pipeline
    rag = RAGPipeline()
    
    if rag.test_connection():
        print("RAG Pipeline initialized successfully!")
        
        # Test with sample questions
        test_questions = [
            "Apa itu Universitas Gunadarma?",
            "Bagaimana cara mendaftar di Universitas Gunadarma?",
            "Fakultas apa saja yang ada di Universitas Gunadarma?"
        ]
        
        for question in test_questions:
            print(f"\nPertanyaan: {question}")
            result = rag.ask_question(question)
            print(f"Status: {result['status']}")
            print(f"Jawaban: {result['answer']}")
            print(f"Sumber: {result['source_urls']}")
            print("-" * 50)
    else:
        print("Failed to initialize RAG Pipeline. Please check your configuration.")