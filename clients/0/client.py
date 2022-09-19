MAIN_FOLDER = 'D:/Me/Codes/Globalink Mitacs/FL-Simplified'
import sys
sys.path.append(MAIN_FOLDER)

import os
import sys
import time

import socket
from fl_simplified.comm.info_exchange import send_json
from fl_simplified.comm.model_exchange import send_model, recv_model

import numpy as np
import torch
from torch.utils.data import DataLoader
from fl_simplified.ml.model_design import Net, train_model, evaluate_model

""" Macros """
HOST = "localhost"
PORT = 9999
BUFFER_SIZE = 1024
REQUEST_INFO_MSG = "Requesting Info"

MODEL_FOLDERNAME = MAIN_FOLDER + '/clients/0/model/'
RECV_MODEL_FILENAME = 'model_recv.pth'
RECV_MODEL_PATH = MODEL_FOLDERNAME + RECV_MODEL_FILENAME
UPDATED_MODEL_FILENAME = 'model_updated.pth'
UPDATED_MODEL_PATH = MODEL_FOLDERNAME + UPDATED_MODEL_FILENAME

X_TRAIN_PATH = MAIN_FOLDER + '/clients/0/data/x_train.npy'
Y_TRAIN_PATH = MAIN_FOLDER + '/clients/0/data/y_train.npy'

X_TEST_PATH = MAIN_FOLDER + '/clients/0/data/x_test.npy'
Y_TEST_PATH = MAIN_FOLDER + '/clients/0/data/y_test.npy'

BATCH_SIZE = 32

def main(latency = 0):

    """ Creating & Connecting a Socket """
    sock_fd = socket.socket()
    sock_fd.connect((HOST, PORT))

    while True:
        """ Receive Request Info Message """
        msg = sock_fd.recv(BUFFER_SIZE).decode('utf-8')
        assert msg == REQUEST_INFO_MSG, f'[Invalid Message Received]: Expecting "{REQUEST_INFO_MSG}"'

        """ Load Dataset into Data Loader """
        X_train = torch.from_numpy(np.load(X_TRAIN_PATH)) / 255
        y_train = torch.from_numpy(np.load(Y_TRAIN_PATH)).type(torch.LongTensor)

        X_test = torch.from_numpy(np.load(X_TEST_PATH)) / 255
        y_test = torch.from_numpy(np.load(Y_TEST_PATH)).type(torch.LongTensor)

        train = torch.utils.data.TensorDataset(X_train,y_train)
        test = torch.utils.data.TensorDataset(X_test,y_test)

        train_dataloader = DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
        test_dataloader = DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

        """ Sending Info """
        info_json = {   "is_idle":True, 
                        "is_charging": True, 
                        "is_connected_to_wifi": True,
                        "training_samples": len(train_dataloader.dataset) }
        send_json(sock_fd=sock_fd, obj=info_json)

        """ Receiving & Saving Model """
        recv_model(sock_fd=sock_fd, to_save_filename=RECV_MODEL_PATH)

        """ Load Model and Setting Loss Function & Optimizer """
        model = torch.load(RECV_MODEL_PATH)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        """ Train Model on Dataset """
        train_model(model=model, train_dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=5, batch_view_dims=(-1, 1, 28, 28))

        """ Saving Model in File """
        torch.save(model, UPDATED_MODEL_PATH)

        """ Testing Model Accuracy """
        evaluate_model(model=model, test_dataloader=test_dataloader)

        """ Wait 5 Seconds before Clearing the Screen """
        time.sleep(5.0)
        os.system('cls')

        """ Adding Latency """
        time.sleep(latency)

        """ Sending Model to Server """
        send_model(sock_fd=sock_fd, to_send_filename=UPDATED_MODEL_PATH)

    sock_fd.close()


if __name__ == "__main__":
    latency = 0
    if len(sys.argv) > 1:
        latency = float(sys.argv[1])
    main(latency=latency)