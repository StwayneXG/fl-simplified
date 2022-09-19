MAIN_FOLDER = 'D:/Me/Codes/Globalink Mitacs/FL-Simplified'
import sys
sys.path.append(MAIN_FOLDER)

from socket import socket
from fl_simplified.comm.info_exchange import send_json, recv_json
from fl_simplified.comm.model_exchange import send_model, recv_model

import torch

REQUEST_INFO_MSG = "Requesting Info"

class ClientConnected:
    def __init__(self, sock_fd: socket, ip: str, port: str):
        self.sock_fd = sock_fd
        self.ip = ip
        self.port = port
        self.info = None

        self.model_statedict = None
        self.model_updated = False
        

    def get_info(self):
        """ Request Info """
        self.sock_fd.send(bytes(REQUEST_INFO_MSG, encoding="utf-8"))

        """ Getting Client Data """
        self.info = recv_json(sock_fd=self.sock_fd)


    def exchange_model(self, to_send_filename: str, to_save_filename: str):
        """ Sending Model to Client """
        send_model(sock_fd=self.sock_fd, to_send_filename=to_send_filename)
        self.model_updated = False

        """ Receiving Updated Model from Client """
        recv_model(sock_fd=self.sock_fd, to_save_filename=to_save_filename)
        self.model_updated = True

        """ Saving Model in Client Object """
        model = torch.load(to_save_filename)
        self.model_sd = model.state_dict()
        