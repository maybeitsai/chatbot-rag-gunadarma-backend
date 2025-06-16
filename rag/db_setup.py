import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

def setup_database():
    """Setup PGVector database and create necessary extensions"""
    connection_string = os.getenv("NEON_CONNECTION_STRING")
    
    try:
        # Connect to database
        conn = psycopg2.connect(connection_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create pgvector extension if not exists
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("PGVector extension created/verified successfully")
        
        # Drop existing tables if they exist
        cursor.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE;")
        print("Existing tables dropped successfully")
        
        cursor.close()
        conn.close()
        print("Database setup completed successfully")
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        raise

if __name__ == "__main__":
    setup_database()