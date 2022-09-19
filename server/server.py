MAIN_FOLDER = 'D:/Me/Codes/Globalink Mitacs/FL-Simplified'
import sys
sys.path.append(MAIN_FOLDER)

import os
from glob import glob
import re

import socket
from fl_simplified.for_server.client_design import ClientConnected
from fl_simplified.for_server.utils import initialize_global_model, delete_all_model_files

from threading import Thread, Event
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from fl_simplified.ml.model_design import evaluate_model


""" Macros """
HOST = "localhost"
PORT = 9999
MAX_CLIENTS = 10

MODELS_FOLDER = MAIN_FOLDER + "/server/models/"
GLOBAL_MODEL_FILENAME = "model_global.pth"
GLOBAL_MODEL_PATH = MODELS_FOLDER + GLOBAL_MODEL_FILENAME

X_TEST_PATH = MAIN_FOLDER + '/server/data/x_test.npy'
Y_TEST_PATH = MAIN_FOLDER + '/server/data/y_test.npy'
BATCH_SIZE = 64

TIMEOUT_DELAY = 10
AGGREGATION_WAIT_TIME = 30
MINIMUM_CLIENTS = 1

""" Global Variables """
global_model = None
clients = list()
isdone_aggregating = Event()

def main(aggregation_waitcycle, timeout_delay):
    global global_model
    global_model = initialize_global_model(GLOBAL_MODEL_PATH)

    """ Creating Listening Socket """
    server = socket.socket()
    server.bind((HOST, PORT))
    server.listen(MAX_CLIENTS)

    """ Start Listening Thread """
    Thread(target=listening_thread, args=(server,), daemon=True).start()

    """ Start Aggregation Thread """
    Thread(target=aggregation_thread, args=(aggregation_waitcycle, timeout_delay,), daemon=True).start()
    
    input("[Press Enter to Exit Server]\n")
    

def aggregation_thread(aggregation_waitcycle, timeout_delay):
    X_test = torch.from_numpy(np.load(X_TEST_PATH)) / 255
    y_test = torch.from_numpy(np.load(Y_TEST_PATH)).type(torch.LongTensor)
    test = torch.utils.data.TensorDataset(X_test,y_test)
    test_dataloader = DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
    
    global global_model
    while True:
        time.sleep(aggregation_waitcycle)
        for client in clients:
            if not client.model_updated:
                print(f"Waiting {timeout_delay} Seconds for Clients to Send Models. (Not Included in Aggregation if Client Doesn't Send Model)")
                time.sleep(timeout_delay)
                break
        total_size = 0
        involved_clients = 0
        for client in clients:
            if client.model_updated:
                involved_clients += 1
                total_size += client.info['training_samples']        

        if involved_clients >= MINIMUM_CLIENTS:
            global_model_sd = global_model.state_dict()
            
            for key in global_model_sd:
                global_model_sd[key] = 0
                for client in clients:
                    if client.model_updated:
                        global_model_sd[key] = global_model_sd[key] + (client.model_sd[key] * client.info['training_samples']) 
                global_model_sd[key] /= total_size

            global_model.load_state_dict(global_model_sd)
            torch.save(global_model, GLOBAL_MODEL_PATH)

            """ Delete All Models """
            delete_all_model_files(models_folder=MODELS_FOLDER)

            print("[Aggregation Complete]")
            print(f"Clients Involved: {involved_clients}")
            evaluate_model(model=global_model, test_dataloader=test_dataloader, batch_view_dims=(-1, 1, 28, 28))
        else:
            print(f"Not enough clients ready, Currently ready = {involved_clients}, Required = {MINIMUM_CLIENTS}")
        isdone_aggregating.set()
        time.sleep(1)
        isdone_aggregating.clear()


""" Daemon Thread Used to Listen For New Connections """
""" args:
        listener_sock = Listening Socket of Server """
""" return: None """
def listening_thread(listener_sock: socket.socket):
    while True:
        client_sock, address = listener_sock.accept()
        ip, port = str(address[0]), str(address[1])
        client = ClientConnected(sock_fd=client_sock, ip=ip, port=port)
        clients.append(client)

        Thread(target=client_thread, args=(client,)).start()
    listener_sock.close()


""" Client Thread for Each New Client """
""" args:
        client = Newly Connected Client """
""" return: None """
def client_thread(client: ClientConnected):
    
    while True:
        """ Get Info """
        client.get_info()

        """ Exchange Models """
        recv_model_path = MODELS_FOLDER+f"model_client({client.port})({client.info['training_samples']}).pth"
        client.exchange_model(to_send_filename=GLOBAL_MODEL_PATH, to_save_filename=recv_model_path)

        """ Await Aggregation """
        isdone_aggregating.wait()

        """ Signal for Client Model Received """
    client.sock_fd.close()



if __name__ == "__main__":
    aggregation_waitcycle = AGGREGATION_WAIT_TIME
    timeout_delay = TIMEOUT_DELAY
    if len(sys.argv) > 1:
        aggregation_waitcycle = float(sys.argv[1])
        if len(sys.argv) > 2:
            latency = float(sys.argv[2])
    main(aggregation_waitcycle=aggregation_waitcycle, timeout_delay=timeout_delay, )
