import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

def setup_database():
    """Setup PGVector database and create necessary extensions"""
    connection_string = os.getenv("NEON_CONNECTION_STRING")
    
    if not connection_string:
        raise ValueError("NEON_CONNECTION_STRING environment variable is not set")
    
    try:
        # Connect to database
        conn = psycopg2.connect(connection_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create pgvector extension if not exists
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✅ PGVector extension created/verified successfully")
        
        # Drop existing tables if they exist (for clean setup)
        cursor.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE;")
        print("🗑️  Existing tables dropped successfully")
        
        # Create enhanced metadata table for better content management
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_metadata (
                id SERIAL PRIMARY KEY,
                url VARCHAR(2048) UNIQUE NOT NULL,
                title TEXT,
                content_hash VARCHAR(64),
                source_type VARCHAR(50),
                domain VARCHAR(255),
                crawl_date DATE,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content_length INTEGER,
                chunk_count INTEGER DEFAULT 0,
                update_reason VARCHAR(100),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("📋 Content metadata table created successfully")
        
        # Create index for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_metadata_url ON content_metadata(url);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_metadata_domain ON content_metadata(domain);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_metadata_hash ON content_metadata(content_hash);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_metadata_date ON content_metadata(crawl_date);")
        print("🔍 Indexes created successfully")
        
        cursor.close()
        conn.close()
        print("✅ Database setup completed successfully")
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        raise

def reset_database():
    """Reset database by dropping all tables and recreating them"""
    connection_string = os.getenv("NEON_CONNECTION_STRING")
    
    if not connection_string:
        raise ValueError("NEON_CONNECTION_STRING environment variable is not set")
    
    try:
        conn = psycopg2.connect(connection_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Drop all related tables
        cursor.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS content_metadata CASCADE;")
        print("🗑️  All tables dropped successfully")
        
        cursor.close()
        conn.close()
        
        # Recreate database
        setup_database()
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        raise

def check_database_status():
    """Check database connection and table status"""
    connection_string = os.getenv("NEON_CONNECTION_STRING")
    
    if not connection_string:
        print("❌ NEON_CONNECTION_STRING environment variable is not set")
        return False
    
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        # Check pgvector extension
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
        vector_exists = cursor.fetchone()[0]
        
        # Check tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('langchain_pg_embedding', 'langchain_pg_collection', 'content_metadata')
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get table row counts
        table_counts = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                table_counts[table] = cursor.fetchone()[0]
            except:
                table_counts[table] = "Error"
        
        cursor.close()
        conn.close()
        
        print("📊 DATABASE STATUS")
        print("=" * 40)
        print(f"PGVector extension: {'✅ Installed' if vector_exists else '❌ Not installed'}")
        print(f"Connection: ✅ Success")
        
        print(f"\nTables:")
        expected_tables = ['langchain_pg_embedding', 'langchain_pg_collection', 'content_metadata']
        for table in expected_tables:
            if table in tables:
                count = table_counts.get(table, 0)
                print(f"  {table}: ✅ Exists ({count} rows)")
            else:
                print(f"  {table}: ❌ Missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "setup":
            setup_database()
        elif command == "reset":
            reset_database()
        elif command == "status":
            check_database_status()
        else:
            print("Usage: python db_setup.py [setup|reset|status]")
    else:
        print("🚀 Starting database setup...")
        setup_database()
        print("\n📊 Checking status...")
        check_database_status()