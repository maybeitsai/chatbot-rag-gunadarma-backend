from scripts.src.setup.core.config import SetupConfig
from scripts.src.setup.core.enums import SetupStep
from scripts.src.setup.core.logger import Logger
from scripts.src.setup.core.tracker import StepTracker
from scripts.src.setup.managers.cache_manager import CacheManager
from scripts.src.setup.managers.data_crawler import DataCrawler
from scripts.src.setup.managers.data_processor import DataProcessor
from scripts.src.setup.managers.system_tester import SystemTester
from scripts.src.setup.validators.environment import EnvironmentValidator
from scripts.src.setup.ui.display import Display

class RAGSystemSetup:
    def __init__(self, config: SetupConfig):
        self.config = config
        self.logger = Logger.setup_logging(config.log_level)
        self.tracker = StepTracker(self.logger)
        self.cache_manager = CacheManager(self.logger)
        self.data_crawler = DataCrawler(self.logger, self.tracker)
        self.data_processor = DataProcessor(self.logger, self.tracker)
        self.system_tester = SystemTester(self.logger, self.tracker)

    async def run_complete_setup(self) -> bool:
        Display.show_welcome_banner()
        Display.show_configuration(self.config)

        try:
            if not self.check_environment():
                return False

            if not self.setup_database():
                return False

            Display.check_cache_status(self.cache_manager)

            if not self.config.skip_crawling:
                if not await self.data_crawler.crawl_data(self.config.force_crawl):
                    return False
            else:
                Display.skip_crawling()

            if not self.data_processor.process_and_optimize_data():
                return False

            if not self.system_tester.test_vector_store():
                return False

            if not self.system_tester.test_system():
                return False

            self.optimize_vector_store()
            Display.show_success_message()
            return True

        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            return False
        finally:
            self.tracker.print_summary()

    def check_environment(self) -> bool:
        Display.check_environment_status()
        is_valid, missing_vars = EnvironmentValidator.validate()

        if not is_valid:
            self.tracker.log_step(SetupStep.ENVIRONMENT, False, f"Missing variables: {', '.join(missing_vars)}")
            return False

        self.tracker.log_step(SetupStep.ENVIRONMENT, True, "All required variables found")
        return True

    def setup_database(self) -> bool:
        try:
            from scripts.src.setup.db_setup import setup_database
            setup_database()
            self.tracker.log_step(SetupStep.DATABASE, True, "Database tables created")
            return True
        except Exception as e:
            self.tracker.log_step(SetupStep.DATABASE, False, str(e))
            return False

    def optimize_vector_store(self) -> bool:
        try:
            from scripts.src.setup.vector_store import VectorStoreManager
            vector_manager = VectorStoreManager()
            vector_manager.cleanup_old_collections(keep_latest=2)
            vector_manager.create_indexes()
            self.tracker.log_step(SetupStep.OPTIMIZATION, True, "Performance optimized")
            return True
        except Exception as e:
            self.tracker.log_step(SetupStep.OPTIMIZATION, False, str(e))
            return False