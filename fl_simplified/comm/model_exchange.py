import os
import tqdm
import socket


""" Global Variables """
BUFFER_SIZE = 1024
SEPARATOR = "%SZ%"

""" Sending Model File """
""" args:
        sock_fd = Socket Used to Send,
        to_send_filename = Name of Model '.pt' or '.pth' File to Send """
""" return: None """
def send_model(sock_fd: socket.socket, to_send_filename: str):
    """ Get File Info """
    filesize = os.path.getsize(to_send_filename)

    """ Send File Info """
    sock_fd.send(f"{to_send_filename}{SEPARATOR}{filesize}".encode("utf-8"))
  
    """ Progress Bar Sending """
    progress = tqdm.tqdm(range(filesize), f"Sending {to_send_filename}", unit="B", unit_scale=True, unit_divisor=BUFFER_SIZE)
    with open(to_send_filename, 'rb') as fr_model:
        while True:
            data = fr_model.read(BUFFER_SIZE)
            if not data:
                break
            sock_fd.sendall(data)
            progress.update(len(data))
        fr_model.close()


""" Receving Model File """
""" args:
        sock_fd = Socket Used to Send,
        to_save_filename = Name of Model '.pt' or '.pth' File to Save """
""" return: None """
def recv_model(sock_fd: socket.socket, to_save_filename: str):
    """ Recv File Info """
    recv = sock_fd.recv(BUFFER_SIZE).decode("utf-8")

    """ Parse File Info """
    filename, filesize = recv.split(SEPARATOR)
    filename = os.path.basename(filename)
    filesize = int(filesize)

    """ Progress Bar Receiving """
    progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=BUFFER_SIZE)
    with open(to_save_filename, 'wb') as fw_model:
        while True:
            data = sock_fd.recv(BUFFER_SIZE)
            if not data:    
                break
            fw_model.write(data)
            progress.update(len(data))
            if len(data) < BUFFER_SIZE:
                break
        fw_model.close()
