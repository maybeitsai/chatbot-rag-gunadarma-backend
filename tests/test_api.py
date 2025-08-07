#!/usr/bin/env python3
"""
Test script untuk API RAG Pipeline
"""

import requests
import json
import time

import pytest

# API base URL
BASE_URL = "http://localhost:8000"

@pytest.fixture
def sample_questions():
    """Fixture providing sample questions for testing"""
    return [
        "Apa itu Universitas Gunadarma?",
        "Fakultas apa saja yang ada di Universitas Gunadarma?",
        "Bagaimana cara mendaftar di Universitas Gunadarma?",
        "Pertanyaan yang tidak ada jawabannya dalam data"
    ]

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        print("✅ Health check passed")
    except Exception as e:
        print(f"❌ Health check error: {e}")
        raise

def test_root():
    """Test root endpoint"""
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200, f"Root endpoint failed: {response.status_code}"
        data = response.json()
        print(f"✅ Root endpoint: {data['message']}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        raise

def test_stats():
    """Test stats endpoint"""
    print("🔍 Testing stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/stats")
        assert response.status_code == 200, f"Stats endpoint failed: {response.status_code}"
        data = response.json()
        print(f"✅ Stats - LLM: {data.get('llm_model')}, Embedding: {data.get('embedding_model')}")
    except Exception as e:
        print(f"❌ Stats endpoint error: {e}")
        raise

def test_examples():
    """Test examples endpoint"""
    print("🔍 Testing examples endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/examples")
        assert response.status_code == 200, f"Examples endpoint failed: {response.status_code}"
        data = response.json()
        examples = data.get('example_questions', [])
        print(f"✅ Examples endpoint - {len(examples)} example questions")
    except Exception as e:
        print(f"❌ Examples endpoint error: {e}")
        raise

def test_ask_question_basic():
    """Test ask endpoint with a basic question"""
    question = "Apa itu Universitas Gunadarma?"
    print(f"🔍 Testing question: '{question}'")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/ask",
            headers={"Content-Type": "application/json"},
            json={"question": question}
        )
        
        assert response.status_code == 200, f"Ask endpoint failed: {response.status_code}"
        data = response.json()
        assert 'status' in data, "Response missing 'status' field"
        assert 'answer' in data, "Response missing 'answer' field"
        assert 'source_urls' in data, "Response missing 'source_urls' field"
        
        print(f"✅ Status: {data['status']}")
        print(f"📝 Answer: {data['answer'][:100]}...")
        print(f"🔗 Sources: {len(data['source_urls'])} URLs")
        if data['source_urls']:
            print(f"   First source: {data['source_urls'][0]}")
    except Exception as e:
        print(f"❌ Ask endpoint error: {e}")
        raise

@pytest.mark.parametrize("question", [
    "Apa itu Universitas Gunadarma?",
    "Fakultas apa saja yang ada di Universitas Gunadarma?", 
    "Bagaimana cara mendaftar di Universitas Gunadarma?"
])
def test_ask_questions_parametrized(question):
    """Test ask endpoint with parametrized questions"""
    print(f"🔍 Testing question: '{question}'")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/ask",
            headers={"Content-Type": "application/json"},
            json={"question": question}
        )
        
        assert response.status_code == 200, f"Ask endpoint failed: {response.status_code}"
        data = response.json()
        assert 'status' in data, "Response missing 'status' field"
        assert 'answer' in data, "Response missing 'answer' field"
        assert 'source_urls' in data, "Response missing 'source_urls' field"
        
        print(f"✅ Status: {data['status']}")
        print(f"📝 Answer: {data['answer'][:100]}...")
        print(f"🔗 Sources: {len(data['source_urls'])} URLs")
    except Exception as e:
        print(f"❌ Ask endpoint error: {e}")
        raise

