import requests
import json
import sys

def send_message_to_streamlit(message, base_url="http://localhost:8501"):
    """
    Sends a message to the Streamlit server and returns the response
    """
    try:
        # The /session endpoint is needed to initialize a session with Streamlit
        session_response = requests.get(f"{base_url}/session")
        if not session_response.ok:
            return f"Failed to initialize session: {session_response.status_code}"

        # Extract session information
        session_info = json.loads(session_response.text)
        session_id = session_info.get("id")

        # Prepare headers for subsequent requests
        headers = {
            "Content-Type": "application/json",
            "X-Streamlit-Client": "streamlit-client",
        }

        # Prepare the message data
        data = {
            "session_id": session_id,
            "message": message,
        }

        # Send the message to the Streamlit server
        response = requests.post(
            f"{base_url}/_stcore/message",
            headers=headers,
            json=data
        )

        if response.ok:
            return response.json().get("response", "No response received")
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Streamlit server. Make sure it's running on http://localhost:8501"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("Streamlit CLI Client")
    print("Type 'exit' to quit")
    print("-" * 50)

    while True:
        # Get user input
        user_input = input("\nEnter your message: ").strip()

        # Check if user wants to exit
        if user_input.lower() == 'exit':
            print("Goodbye!")
            sys.exit(0)

        # Send message and get response
        response = send_message_to_streamlit(user_input)
        
        # Print the response
        print("\nServer Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    main()
