import socket


class Receiver:

    def __init__(self):
        super().__init__()
        self.HOST = '10.0.0.102'   # use '' to expose to all networks
        self.PORT = 40

    def receive(self):
        """Open specified port and return file-like object"""
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.HOST, self.PORT))
        sock.listen(0)
        request, addr = sock.accept()
        return request.makefile('r')

# r = Receiver()
# for line in r.receive():
#     print(line)