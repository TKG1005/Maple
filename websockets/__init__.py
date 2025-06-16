class WebSocketClientProtocol:
    pass

class ClientConnection(WebSocketClientProtocol):
    pass

async def connect(*args, **kwargs):
    class Dummy:
        async def __aenter__(self):
            return ClientConnection()
        async def __aexit__(self, exc_type, exc, tb):
            pass
    return Dummy()
