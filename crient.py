import socket

class PokemonClient:

    def __init__(self, client_id, host="localhost", port=8000):
        self.host = host
        self.port = port
        self.client_id = client_id

        # Create a TCP/IP socket
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        # Connect the socket to the server's port
        self.client.connect((self.host, self.port))

        # Send the client ID to the server
        self.client.send(bytes(self.client_id, "utf8"))

        print(f"Client {self.client_id} connected to server {self.host}:{self.port}")

    def send(self, message):
        # Send a message to the server
        self.client.send(bytes(message, "utf8"))

        # Receive a response from the server
        response = self.client.recv(1024).decode("utf8")
        print(f"Received response from server: {response}")

if __name__ == "__main__":
    client_id = "Client1"  # Should be unique for each client
    client = PokemonClient(client_id)
    client.connect()

    while True:
        message = input("Enter message: ")
        client.send(message)
