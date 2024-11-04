import websocket
import requests
import json
import time
import _thread
import rel
import sys

class StreamlitClient:
    def __init__(self, host="localhost", port="8501"):
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/stream"
        self.session_id = None
        self.ws = None
        self.connected = False

    def initialize_session(self):
        """Initialize a session with Streamlit server"""
        try:
            response = requests.get(f"{self.base_url}/_stcore/health")
            if response.status_code == 200:
                return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to Streamlit server: {e}")
            return False

    def on_message(self, ws, message):
        """Handle incoming websocket messages"""
        try:
            data = json.loads(message)
            if 'msg' in data:
                if data['msg'] == 'server_info':
                    self.session_id = data['session_id']
                    self.connected = True
                elif data['msg'] == 'write':
                    # Extract and print the actual message content
                    if 'data' in data and 'text' in data['data']:
                        print("\nServer Response:")
                        print("-" * 50)
                        print(data['data']['text'])
                        print("-" * 50)
        except json.JSONDecodeError:
            print("Received invalid JSON message")

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.connected = False
        print("### Connection closed ###")

    def on_open(self, ws):
        print("Connected to Streamlit server")
        # Send initial configuration
        ws.send(json.dumps({
            "config": {
                "client": "streamlit-client",
                "theme": "light",
                "sessionId": None
            }
        }))

    def connect(self):
        """Establish websocket connection"""
        if not self.initialize_session():
            print("Failed to initialize session with Streamlit server")
            return False

        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        _thread.start_new_thread(self.ws.run_forever, ())
        
        # Wait for connection to be established
        timeout = 5
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        return self.connected

    def send_message(self, message):
        """Send a message to the Streamlit server"""
        if not self.connected or not self.ws:
            print("Not connected to server")
            return False

        try:
            self.ws.send(json.dumps({
                "msg": "script_changed",
                "data": message
            }))
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False

    def close(self):
        """Close the websocket connection"""
        if self.ws:
            self.ws.close()

def main():
    print("Streamlit CLI Client")
    print("Type 'exit' to quit")
    print("-" * 50)

    client = StreamlitClient()
    
    if not client.connect():
        print("Failed to connect to Streamlit server")
        sys.exit(1)

    try:
        while True:
            user_input = input("\nEnter your message: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
                
            if not client.send_message(user_input):
                print("Failed to send message. Reconnecting...")
                if not client.connect():
                    print("Failed to reconnect. Exiting...")
                    break
    finally:
        client.close()

if __name__ == "__main__":
    main()
