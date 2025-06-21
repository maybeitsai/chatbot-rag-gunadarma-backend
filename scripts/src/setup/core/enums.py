from enum import Enum


class SetupStep(Enum):
    ENVIRONMENT = "Environment Check"
    DATABASE = "Database Setup"
    CACHE_STATUS = "Cache Status"
    DATA_CRAWL = "Data Crawling"
    DATA_PROCESS = "Data Processing"
    VECTOR_TEST = "Vector Store Test"
    SYSTEM_TEST = "System Test"
    OPTIMIZATION = "Vector Store Optimization"
