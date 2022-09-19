MAIN_FOLDER = 'D:/Me/Codes/Globalink Mitacs/FL-Simplified'
import sys
sys.path.append(MAIN_FOLDER)

from glob import glob
import os

import torch
from fl_simplified.ml.model_design import Net

""" Check if Global Model File Exists (If not, Create one) """
""" args: 
        global_modelpath = Path of Model """
""" return: Global Model (newly created or loaded from file) """
def initialize_global_model(global_modelpath: str):
    if os.path.exists(global_modelpath):
        model = torch.load(global_modelpath)
        return model
    model = Net()
    torch.save(model, global_modelpath)
    return model


""" Delete All Model Files in Folder """
""" args: 
        folder = Folder Containing Model Files """
""" return: None """
def delete_all_model_files(models_folder: str):
    model_filenames = glob(models_folder + "model_client(*)(*).pth")

    for filename in model_filenames:
        os.remove(filename)
