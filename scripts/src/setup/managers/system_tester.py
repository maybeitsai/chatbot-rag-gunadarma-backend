from rich.console import Console
from rich.status import Status
from scripts.src.setup.core.tracker import StepTracker, SetupStep
from typing import Optional, List, Dict, Any
import logging


class SystemTester:
    """System testing functionality with Rich output"""

    def __init__(self, logger: logging.Logger, tracker: StepTracker):
        self.logger = logger
        self.tracker = tracker

    def test_vector_store(self) -> bool:
        """Test vector store functionality"""
        try:
            from app.rag.vector_store import VectorStoreManager

            with Status("🧪 Testing vector store...", console=Console()):
                vector_manager = VectorStoreManager()
                stats = vector_manager.get_vector_store_stats()

            if stats.get("document_count", 0) == 0:
                self.tracker.log_step(
                    SetupStep.VECTOR_TEST, False, "No documents in vector store"
                )
                return False

            # Test retriever and vector store initialization
            with Status("🔍 Testing retriever...", console=Console()):
                retriever = vector_manager.get_retriever(
                    search_type="similarity_score_threshold", k=3, score_threshold=0.3
                )

            if not retriever:
                self.tracker.log_step(
                    SetupStep.VECTOR_TEST, False, "Failed to initialize retriever"
                )
                return False

            with Status("⚡ Testing vector store initialization...", console=Console()):
                vector_store = vector_manager.initialize_vector_store()

            if not vector_store:
                self.tracker.log_step(
                    SetupStep.VECTOR_TEST, False, "Failed to initialize vector store"
                )
                return False

            self.tracker.log_step(
                SetupStep.VECTOR_TEST,
                True,
                f"Vector store working with {stats['document_count']} documents",
            )
            return True

        except Exception as e:
            self.tracker.log_step(SetupStep.VECTOR_TEST, False, str(e))
            return False

    def test_system(self) -> bool:
        """Test complete system"""
        try:
            from app.rag.pipeline import create_rag_pipeline

            with Status("🔗 Creating RAG pipeline...", console=Console()):
                pipeline = create_rag_pipeline(enable_cache=True)

            # Test connection if available
            if hasattr(pipeline, "test_connection"):
                with Status("🔌 Testing connection...", console=Console()):
                    if not pipeline.test_connection():
                        self.tracker.log_step(
                            SetupStep.SYSTEM_TEST, False, "Connection test failed"
                        )
                        return False

            # Test with sample question
            test_question = "Apa itu Universitas Gunadarma?"
            with Status(
                f"❓ Testing with question: '{test_question}'...", console=Console()
            ):
                result = pipeline.ask_question(test_question)

            if result and result.get("status") in ["success", "not_found"]:
                self.tracker.log_step(
                    SetupStep.SYSTEM_TEST,
                    True,
                    f"Test question processed: {result.get('status')}",
                )
                return True
            else:
                self.tracker.log_step(
                    SetupStep.SYSTEM_TEST, False, "Failed to process test question"
                )
                return False

        except Exception as e:
            self.tracker.log_step(SetupStep.SYSTEM_TEST, False, str(e))
            return False
