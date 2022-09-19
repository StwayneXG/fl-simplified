# Federated Learning Simplified

This repository focuses on setting up an infrastructure of Federated Learning which could be used and expanded to real life applications (NOT SIMULATIONS).

### Brief Introduction

Federated Learning is the technique of machine learning used to train models on edge devices rather than on cloud or a central server. This technique was proposed to highlight the problem of data privacy. People are reluctant to provide their data to companies as it could potentially have private information. This would be a huge barrier in the field of machine learning as all of it revolves around abundance of data. Federated Learning is a great solution for that as it provides insurance that no data is leaked from their devices to any company.

## Communication

The communication part is implemented using standard socket library in Python used for the exchange of machine learning models. This part is still being worked on to make the server more robust. Events & Signals have been used to communicate between threads and cooperate with each other.

## Machine Learning

The primary objective of this project was to develop the environment to make federated learning possible over the network. So, the machine learning part included a simple CNN model for classification. The model can be edited in the ``` ./fl_simplified/ml/model_design.py ``` file. Any machine learning model should work for this repo.

## Dataset

Federated Learning requires a non IID dataset. However, for this project, I just used partitions of MNIST dataset available online. A non IID dataset can be found online on https://github.com/TalwalkarLab/leaf provided by LEAF which is an excellent benchmark for Federated Learning

## Current Objectives

Currently, the problems I've been facing are:
* Killing the client thread immediately if it has not send the model before time out. (I've looked online for a couple of solutions but they run into errors and bugs and require redesign for the server).
* Delaying any new client that connects during the aggregation process. Making it wait for a newer/better global to be sent.

For further work:
* If there is an agreement between client and server to use a specific model before connection or at the start. Best practice would be to share model parameters only, without the design as it is much safer and robust.
* The goal of Federated Learning is to prioritize data privacy by training the model over edge devices. But studies have shown, it is possible to extract the data a model was trained upon using the trained model. I plan to add homomorphic encryption at client side to further enhance data privacy.

## Instructions

#### Before starting:

Change the value of MAIN_FOLDER macro at the start of ``` ./clients/0/client.py ```, ``` ./server/server.py ```, ``` ./fl_simplified/for_server/client_design.py ```, ``` ./fl_simplified/for_server/utils.py ``` to the folder you clone this too.

#### To Run:

Run the server.py by using ``` python server.py AGGREGATION_CYCLETIME TIMEOUT_DELAY ```\
Where ```AGGREGATION_CYCLETIME``` is the time in seconds between each aggregation round,\
and ```TIMEOUT_DELAY``` is the extra time (in seconds) given to clients to reconnect and send the model to server before being removed from the aggregation round.

Run the client.py by using ``` python client.py LATENCY ```\
Where ```LATENCY``` is the time in seconds added as an artificial latency before sending model to server.

### More Info

The communication in this can adapt as many clients as possible. Run the ```client.py``` file as many times with different dataset to test out Federated Learning accuracy for multiple clients. (I will create a config file for easier execution of multiple clients)

## License

[MIT](LICENSE) © Irtaza Sajid Qureshi
