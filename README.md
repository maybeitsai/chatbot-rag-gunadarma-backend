# Gunadarma RAG Pipeline

Pipeline Retrieval-Augmented Generation (RAG) untuk informasi Universitas Gunadarma menggunakan LangChain, PGVector, dan Google Gemini.

## Fitur

- ✅ Menggunakan LangChain sebagai framework utama
- ✅ PGVector dengan Neon DB untuk vector store
- ✅ Google Gemini untuk LLM dan embedding
- ✅ FastAPI untuk REST API
- ✅ Docker support untuk deployment
- ✅ Chunking dengan konfigurasi CHUNK_SIZE=500, CHUNK_OVERLAP=50
- ✅ Hanya menjawab berdasarkan dokumen yang tersedia
- ✅ Menyertakan URL sumber dalam setiap jawaban

## Struktur Project

```
.
└── data/
    ├── output.json
    └── output.csv
└── rag/  
    ├── pipeline.py
    ├── vector_store.py 
    ├── data_processor.py
    ├── db_setup.py
├── main.py
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── crawler.log
├── uv.lock
├── run.sh
├── .env 
├── .python-version
├── .gitignore

```

## Instalasi

### 1. Persiapan Environment

```bash
# Clone
git clone https://github.com/maybeitsai/chatbot-rag-gunadarma-backend.git
cd chatbot-rag-gunadarma-backend

# Install uv (jika belum ada)
curl -LsSf https://astral.sh/uv/install.sh | less

# Install dependencies
uv sync
```

### 2. Konfigurasi Environment Variables

Buat file `.env` dengan konfigurasi berikut:

```env
GOOGLE_API_KEY=your-api-gemini
NEON_CONNECTION_STRING=your-api-neon
CHUNK_SIZE=500
CHUNK_OVERLAP=50
EMBEDDING_MODEL=models/text-embedding-004
LLM_MODEL=gemini-2.5-flash-preview-05-20
```

### 3. Persiapan Data

Pastikan salah satu dari file data berikut tersedia:
- `data/output.json`
- `data/output.csv`

Format data harus mengandung kolom:
- `url`: URL sumber
- `title`: Judul halaman
- `text_content`: Konten teks
- `source_type`: Tipe sumber (html dan pdf)
- `timestamp`: Waktu crawling

### 4. Setup Database dan Vector Store

```bash
# Jalankan setup script
uv run setup.py
```

Script ini akan:
- Crawling data
- Setup database PGVector
- Hapus data lama (jika ada)
- Process dan chunk dokumen
- Populate vector store dengan embedding

### 5. Menjalankan API

```bash
# Development mode
uv run main.py

# Production mode dengan uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Penggunaan Docker

### Build dan Run dengan Docker Compose

```bash
# Build dan run
docker-compose up --build

# Run di background
docker-compose up -d

# Stop services
docker-compose down
```

### Build Docker Image Manual

```bash
# Build image
docker build -t chatbot-rag-gunadarma-backend .

# Run container
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data:ro chatbot-rag-gunadarma-backend
```

## API Endpoints

### 1. Health Check
```
GET /health
```
Mengecek status kesehatan API dan koneksi database.

### 2. Ask Question (Main Endpoint)
```
POST /ask
Content-Type: application/json

{
  "question": "Apa itu Universitas Gunadarma?"
}
```

Response:
```json
{
  "answer": "Universitas Gunadarma adalah...",
  "source_urls": ["https://www.gunadarma.ac.id/..."],
  "status": "success",
  "source_count": 3
}
```

Status values:
- `success`: Informasi ditemukan dan dijawab
- `not_found`: Informasi tidak tersedia dalam data
- `error`: Terjadi kesalahan sistem

### 3. System Statistics
```
GET /stats
```
Menampilkan statistik sistem dan konfigurasi.

### 4. Example Questions
```
GET /examples
```
Menampilkan contoh pertanyaan yang bisa diajukan.

### 5. Root Endpoint
```
GET /
```
Informasi dasar API.

## Contoh Penggunaan

### Menggunakan curl

```bash
# Health check
curl http://localhost:8000/health

# Ask question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Fakultas apa saja yang ada di Universitas Gunadarma?"}'

# Get examples
curl http://localhost:8000/examples
```

### Menggunakan Python

```python
import requests

# Ask question
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "Bagaimana cara mendaftar di Universitas Gunadarma?"}
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['source_urls']}")
print(f"Status: {result['status']}")
```

## Konfigurasi

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | - | Google API key untuk Gemini |
| `NEON_CONNECTION_STRING` | - | Connection string ke Neon DB |
| `CHUNK_SIZE` | 500 | Ukuran chunk dokumen |
| `CHUNK_OVERLAP` | 50 | Overlap antar chunk |
| `EMBEDDING_MODEL` | models/text-embedding-004 | Model embedding Google |
| `LLM_MODEL` | gemini-2.5-flash-preview-05-20 | Model LLM Google |
| `PORT` | 8000 | Port untuk FastAPI |

### Prompt Engineering

Pipeline menggunakan prompt yang ketat untuk memastikan:
1. Hanya menjawab berdasarkan dokumen sumber
2. Menolak menjawab jika informasi tidak tersedia
3. Selalu menyertakan URL sumber
4. Menjawab dalam bahasa Indonesia

## Troubleshooting

### Database Connection Issues

```bash
# Test database connection
python -c "
import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()
conn = psycopg2.connect(os.getenv('NEON_CONNECTION_STRING'))
print('Database connection successful!')
conn.close()
"
```

### Vector Store Issues

```bash
# Reset vector store
uv run db_setup.py
uv run vector_store.py
```

### API Issues

```bash
# Check logs
docker-compose logs rag-api

# Test individual components
python -c "from pipeline import RAGPipeline; rag = RAGPipeline(); print(rag.test_connection())"
```

## Monitoring

### Health Check Endpoint

API menyediakan endpoint `/health` yang mengecek:
- Status RAG pipeline
- Koneksi database
- Koneksi ke vector store

### Docker Health Check

Container Docker memiliki built-in health check yang memantau endpoint `/health`.

## Keamanan

- API key Google disimpan dalam environment variables
- Database connection menggunakan SSL
- CORS dikonfigurasi untuk production
- Input validation pada semua endpoint

## Limitasi

- Hanya menjawab berdasarkan data yang di-crawl
- Tidak menyimpan riwayat percakapan
- Batasan rate limiting bergantung pada Google API
- Performa bergantung pada ukuran dataset

## Development

### Testing

```bash
# Test individual components
uv run data_processor.py
uv run vector_store.py
uv run pipeline.py

# Test API
uv run main.py
```

### Adding New Features

1. Modifikasi model Pydantic di `main.py`
2. Update RAG pipeline di `pipeline.py`
3. Test dengan endpoint baru
4. Update dokumentasi

## Deployment Production

### Recommendations

1. Gunakan reverse proxy (nginx)
2. Setup monitoring (prometheus/grafana)
3. Implement caching (Redis)
4. Setup backup database
5. Configure log aggregation
6. Use secrets management

### Example nginx config

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.