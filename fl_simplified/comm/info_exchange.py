import socket
import json

""" Global Variables """
BUFFER_SIZE = 1024


""" Sending JSON """
""" args:
        sock_fd = Socket Used to Send,
        obj = Dictionary Containing Information """
""" return: None """
def send_json(sock_fd: socket.socket, obj: dict):

    data = json.dumps(obj)
    sock_fd.send(bytes(data, encoding="utf-8"))


""" Receiving JSON """
""" args: 
        sock_fd = Socket Used to Recv """
""" return: obj = Dictionary Containing Information """
def recv_json(sock_fd: socket.socket):

    obj = sock_fd.recv(BUFFER_SIZE)
    obj = json.loads(obj.decode("utf-8"))

    return obj