#!/usr/bin/env python3
"""
Test script for the enhanced Multi-Agent System
"""

import asyncio
import aiohttp
import json

async def test_system():
    """Test the enhanced system with a banana webpage request"""
    
    print("🧪 Testing Enhanced Multi-Agent System")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Check system status
        print("1. Checking system status...")
        try:
            async with session.get(f"{base_url}/api/system-status") as response:
                if response.status == 200:
                    status = await response.json()
                    print(f"   ✅ System active - {status.get('agents', {}).get('total', 0)} agents")
                else:
                    print(f"   ❌ System status check failed: {response.status}")
        except Exception as e:
            print(f"   ❌ Connection failed: {e}")
            return
        
        # Test 2: Process a banana webpage request
        print("2. Requesting ASCII banana webpage...")
        try:
            request_data = {
                "message": "create a html webpage that creates an ascii banana"
            }
            
            async with session.post(
                f"{base_url}/api/process-request",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✅ Request processed successfully!")
                    print(f"   📋 Tasks created: {result.get('result', {}).get('tasks_created', 0)}")
                    print(f"   🤖 Active agents: {result.get('system_status', {}).get('agents', {}).get('active', 0)}")
                else:
                    print(f"   ❌ Request failed: {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
        
        # Test 3: Check for created files
        print("3. Checking for created files...")
        await asyncio.sleep(5)  # Give agents time to work
        
        try:
            # Try to get the HTML file that should have been created
            async with session.get(f"{base_url}/api/files/html_webpage.html") as response:
                if response.status == 200:
                    file_content = await response.json()
                    content = file_content.get("content", "")
                    if "banana" in content.lower():
                        print("   ✅ ASCII banana webpage created successfully!")
                    else:
                        print("   ⚠️  HTML file created but no banana content found")
                else:
                    print(f"   ⚠️  HTML file not found yet (status: {response.status})")
        except Exception as e:
            print(f"   ❌ File check failed: {e}")
        
        # Test 4: Final system status
        print("4. Final system status...")
        try:
            async with session.get(f"{base_url}/api/system-status") as response:
                if response.status == 200:
                    status = await response.json()
                    agents = status.get('agents', {})
                    tasks = status.get('tasks', {})
                    
                    print(f"   🤖 Agents: {agents.get('active', 0)} active, {agents.get('total', 0)} total")
                    print(f"   📋 Tasks: {tasks.get('completed', 0)} completed, {tasks.get('in_progress', 0)} in progress")
                    
                    if tasks.get('completed', 0) > 0:
                        print("   ✅ System working correctly!")
                    else:
                        print("   ⚠️  No tasks completed yet")
        except Exception as e:
            print(f"   ❌ Status check failed: {e}")
    
    print("\n🎯 Test completed!")
    print("Try accessing http://localhost:8000 to see the interface")

if __name__ == "__main__":
    asyncio.run(test_system())
