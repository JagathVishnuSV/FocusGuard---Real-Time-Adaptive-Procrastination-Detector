#!/usr/bin/env python3
"""
FocusGuard API Test Script
Tests all backend endpoints to verify they're working correctly
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(endpoint, method="GET", data=None):
    """Test a single API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*60}")
    print(f"Testing: {method} {endpoint}")
    print(f"Full URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                json_data = response.json()
                print(f"Response: {json.dumps(json_data, indent=2)}")
                return True, json_data
            except:
                print(f"Response (text): {response.text}")
                return True, response.text
        else:
            print(f"Error: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"Connection Error: {e}")
        return False, None

def main():
    """Test all API endpoints"""
    print("🧪 FocusGuard API Test Suite")
    print("="*60)
    
    # Test basic health check
    print("\n🏥 HEALTH CHECK")
    success, data = test_endpoint("/health")
    if not success:
        print("❌ Backend is not responding! Check if server is running.")
        return
    
    # Test statistics endpoints
    print("\n📊 STATISTICS ENDPOINTS")
    test_endpoint("/api/stats/today")
    test_endpoint("/api/stats/weekly") 
    test_endpoint("/api/stats/hourly")
    
    # Test session management
    print("\n🎯 SESSION MANAGEMENT")
    test_endpoint("/api/session/status")
    
    # Test session start
    print("\n▶️ STARTING SESSION")
    success, session_data = test_endpoint("/api/session/start", "POST")
    
    if success:
        time.sleep(2)  # Let session run for a moment
        
        # Check session status after start
        print("\n📋 SESSION STATUS AFTER START")
        test_endpoint("/api/session/status")
        
        # Test activity feed
        print("\n📱 ACTIVITY FEED")
        test_endpoint("/api/activity/recent")
        
        # Stop session
        print("\n⏹️ STOPPING SESSION")
        test_endpoint("/api/session/stop", "POST")
        
        # Check final status
        print("\n📋 FINAL SESSION STATUS")
        test_endpoint("/api/session/status")
    
    # Test insights and analysis
    print("\n🧠 INSIGHTS & ANALYSIS")
    test_endpoint("/api/insights")
    test_endpoint("/api/distractions/top")
    test_endpoint("/api/features/importance")
    
    # Test export
    print("\n💾 DATA EXPORT")
    test_endpoint("/api/export")
    
    print("\n" + "="*60)
    print("✅ API Test Complete!")
    print("\nIf all endpoints returned 200 status codes, the backend is working correctly.")
    print("Check the frontend at: http://localhost:3001")

if __name__ == "__main__":
    main()
