"""
Description: This module contains the functions to log the accuracy and loss of clients and servers.

Author: Nisal Hemadasa
Date: 10-01-2025
Version: 1.0
"""
import csv
import pickle
from typing import List, Tuple

import constants


def write_logs(loss_and_accuracy: List[any], file_name: str) -> None:
    """
    Plot the loss of the models (of servers or clients) against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the loss and accuracy of the models for each client or server
    :param file_name: Name of the file to save the logs
    :return: None
    """

    def save_as_pkl(_loss_and_accuracy: List[any], _file_name: str) -> None:
        """Save the loss and accuracy data to a binary file"""
        import os
        _file_name = _file_name + constants.FileExtesions.PKL
        print(f"PKL Path: {_file_name}")
        pkl_path_list = _file_name.split('/')
        pkl_dir_path = ""
        for path_elem in pkl_path_list[:-1]:
            pkl_dir_path += path_elem + "/"
        print(f"PKL directory Path: {pkl_dir_path}")
        if not os.path.exists(pkl_dir_path):
            print(f"Creating PKL direcdirectory Path: {pkl_dir_path}")
            os.makedirs(pkl_dir_path)
        # Save to a binary file
        with open(_file_name, "wb") as file:
            pickle.dump(loss_and_accuracy, file)

    def save_as_csv(_loss_and_accuracy: List[any], _file_name: str) -> None:
        """Save the loss and accuracy data to a CSV file"""
        import os
        _file_name = _file_name + constants.FileExtesions.CSV
        print(f"CSV Path: {_file_name}")
        csv_path_list = _file_name.split('/')
        csv_dir_path = ""
        for path_elem in csv_path_list[:-1]:
            csv_dir_path += path_elem + "/"
        print(f"CSV directory Path: {csv_dir_path}")
        if not os.path.exists(csv_dir_path):
            print(f"Creating CSV direcdirectory Path: {csv_dir_path}")
            os.makedirs(csv_dir_path)
        # Save to a CSV file
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(loss_and_accuracy)

    save_as_pkl(loss_and_accuracy, file_name)
    # save_as_csv(loss_and_accuracy, file_name)


def read_logs(file_name: str) -> List[any]:
    """
    Read the loss and accuracy data from a binary file
    :param file_name: Name of the file to read the logs
    :return: List of tuples containing the loss and accuracy of the models for each client or server
    """

    # Read from a binary file
    with open(file_name, "rb") as file:
        loss_and_accuracy = pickle.load(file)

    return loss_and_accuracy
