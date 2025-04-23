import requests
import time
import sys

def check_server(url, max_retries=3, timeout=5):
    """
    Check if the server is running by making a request to the API
    
    Args:
        url (str): The base URL of the API
        max_retries (int): Maximum number of retry attempts
        timeout (int): Timeout in seconds for the request
        
    Returns:
        bool: True if server is accessible, False otherwise
    """
    print(f"Checking server at {url}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}", timeout=timeout)
            if response.status_code == 200:
                print(f"[OK] Server is running and responding")
                return True
            else:
                print(f"[ERROR] Server responded with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Could not connect to server: {str(e)}")
            
        if attempt < max_retries - 1:
            retry_time = (attempt + 1) * 2
            print(f"Retrying in {retry_time} seconds... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_time)
    
    print(f"[FAILED] Server at {url} is not accessible after {max_retries} attempts")
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_server(sys.argv[1])
    else:
        check_server("http://localhost:5000")
