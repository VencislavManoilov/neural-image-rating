import requests
import sys
import os
import time
from dotenv import load_dotenv # type: ignore

def check_server(url, max_retries=3, wait_time=2):
    """Check if the server is running and responsive at the given URL."""
    print(f"Checking server at {url}...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/labels", timeout=5)
            if response.status_code == 200:
                data = response.json()
                label_count = len(data.get('labels', []))
                print(f"✅ Server is running at {url}")
                print(f"Found {label_count} labels in the dataset")
                return True
            else:
                print(f"❌ Server responded with status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection refused (attempt {i+1}/{max_retries})")
        except requests.exceptions.Timeout:
            print(f"❌ Connection timed out (attempt {i+1}/{max_retries})")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        if i < max_retries - 1:
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    print("\nServer check failed. Please make sure:")
    print("1. The server is running")
    print("2. The server URL is correct")
    print("3. Your network allows connections to the server")
    print("\nTo start training with a different URL:")
    print("python train.py --api-url http://your-server-url:port")
    
    return False

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get URL from command line or environment
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = os.getenv("DATASET_URL", "http://localhost:5000")
    
    success = check_server(url)
    sys.exit(0 if success else 1)
