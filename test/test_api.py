#!/usr/bin/env python3
"""
Test script untuk API RAG Pipeline
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root():
    """Test root endpoint"""
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {data['message']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_stats():
    """Test stats endpoint"""
    print("🔍 Testing stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats - LLM: {data.get('llm_model')}, Embedding: {data.get('embedding_model')}")
            return True
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats endpoint error: {e}")
        return False

def test_examples():
    """Test examples endpoint"""
    print("🔍 Testing examples endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/examples")
        if response.status_code == 200:
            data = response.json()
            examples = data.get('example_questions', [])
            print(f"✅ Examples endpoint - {len(examples)} example questions")
            return True
        else:
            print(f"❌ Examples endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Examples endpoint error: {e}")
        return False

def test_ask_question(question):
    """Test ask endpoint with a question"""
    print(f"🔍 Testing question: '{question}'")
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            headers={"Content-Type": "application/json"},
            json={"question": question}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {data['status']}")
            print(f"📝 Answer: {data['answer'][:100]}...")
            print(f"🔗 Sources: {len(data['source_urls'])} URLs")
            if data['source_urls']:
                print(f"   First source: {data['source_urls'][0]}")
            return True
        else:
            print(f"❌ Ask endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ask endpoint error: {e}")
        return False

def main():
    """Main test function"""
    print("=== Gunadarma RAG API Test Suite ===")
    print(f"Testing API at: {BASE_URL}")
    print()
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/")
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(2)
        print(f"   Attempt {i+1}/10...")
    
    # Test basic endpoints
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Root endpoint
    total_tests += 1
    if test_root():
        tests_passed += 1
    print()
    
    # Test 2: Health endpoint
    total_tests += 1
    if test_health():
        tests_passed += 1
    print()
    
    # Test 3: Stats endpoint
    total_tests += 1
    if test_stats():
        tests_passed += 1
    print()
    
    # Test 4: Examples endpoint
    total_tests += 1
    if test_examples():
        tests_passed += 1
    print()
    
    # Test 5-8: Ask questions
    test_questions = [
        "Apa itu Universitas Gunadarma?",
        "Fakultas apa saja yang ada di Universitas Gunadarma?",
        "Bagaimana cara mendaftar di Universitas Gunadarma?",
        "Pertanyaan yang tidak ada jawabannya dalam data"
    ]
    
    for question in test_questions:
        total_tests += 1
        if test_ask_question(question):
            tests_passed += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())