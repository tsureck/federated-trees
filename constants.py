"""
Description: This constants used in this project

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""


# Directory paths related to datasets
class Paths:
    # path to download and read the datasets
    DATASET = 'data/'

    # path related to saved plots
    PLOT_SAVE_PATH = './plots/saved_plots/'

    # path related to saved logs
    LOG_SAVE_PATH = './logs/saved_logs/'
    
    # saved model path
    MODEL_SAVE_PATH = './root_server_model_20_rounds_CIFAR.pth'


# File extensions
class FileExtesions:
    PKL = '.pkl'
    CSV = '.csv'


# String related to miscellaneous messages
class MiscMessages:
    ACCURACY = "accuracy"


# Names of the dataset
class DatasetNames:
    MNIST = 'MNIST'
    F_MNIST = 'FashionMNIST'
    CIFAR_10 = 'cifar-10-batches-py'


class DatasetFileNames:
    # MNIST dataset file names
    MNIST_TRAIN_IMAGES = 'train-images-idx3-ubyte'
    MNIST_TRAIN_LABELS = 'train-labels-idx1-ubyte'
    MNIST_TEST_IMAGES = 't10k-images-idx3-ubyte'
    MNIST_TEST_LABELS = 't10k-labels-idx1-ubyte'

    # F_MNIST dataset file names
    F_MNIST_TRAIN_IMAGES = 'train-images-idx3-ubyte'
    F_MNIST_TRAIN_LABELS = 'train-labels-idx1-ubyte'
    F_MNIST_TEST_IMAGES = 't10k-images-idx3-ubyte'
    F_MNIST_TEST_LABELS = 't10k-labels-idx1-ubyte'

    def get_train_images(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TRAIN_IMAGES]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TRAIN_IMAGES]
        else:
            pass

    def get_train_labels(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TRAIN_LABELS]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TRAIN_LABELS]
        else:
            pass

    def get_test_images(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TEST_IMAGES]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TEST_IMAGES]
        else:
            pass

    def get_test_labels(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TEST_LABELS]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TEST_LABELS]
        else:
            pass


# Strings related to plots
class Plots:
    NUMBER_OF_ROUNDS = 'Rounds'
    LOSS = 'Loss'
    ACCURACY = 'Accuracy'
    # for client individual loss/accuracy vs rounds plot
    CLIENT_LOSS_VS_ROUNDS_TITLE = 'Loss per Client Across Rounds'
    CLIENT_ACCURACY_VS_ROUNDS_TITLE = 'Accuracy per Client Across Rounds'
    CLIENT_LOSS_VS_ROUNDS_PNG = 'client_loss_vs_rounds.png'
    CLIENT_ACCURACY_VS_ROUNDS_PNG = 'client_accuracy_vs_rounds.png'
    # for server individual loss/accuracy vs rounds plot
    SERVER_LOSS_VS_ROUNDS_TITLE = 'Loss per servers Across Rounds'
    SERVER_ACCURACY_VS_ROUNDS_TITLE = 'Accuracy per server Across Rounds'
    SERVER_LOSS_VS_ROUNDS_PNG = 'server_loss_vs_rounds.png'
    SERVER_ACCURACY_VS_ROUNDS_PNG = 'server_accuracy_vs_rounds.png'
    # for client level average loss/accuracy vs rounds plot
    CLIENT_AVG_LOSS_VS_ROUNDS_TITLE = 'Client Level Average Loss Across Rounds'
    CLIENT_AVG_LOSS_VS_ROUNDS_PNG = 'client_avg_loss_vs_rounds.png'
    CLIENT_AVG_ACCURACY_VS_ROUNDS_TITLE = 'Client Level Average Accuracy Across Rounds'
    CLIENT_AVG_ACCURACY_VS_ROUNDS_PNG = 'client_avg_accuracy_vs_rounds.png'
    # for server level average loss/accuracy vs rounds plot
    SERVER_LEVEL_AVG_LOSS_VS_ROUNDS_TITLE = 'Server Level Average Loss Across Rounds'
    SERVER_LEVEL_AVG_LOSS_VS_ROUNDS_PNG = 'server_level_avg_loss_vs_rounds.png'
    SERVER_OVERALL_AVG_LOSS_VS_ROUNDS_TITLE = 'Server Overall Average Loss Across Rounds'
    SERVEr_OVERALL_AVG_LOSS_VS_ROUNDS_PNG = 'server_overall_avg_loss_vs_rounds.png'
    SERVER_LEVEL_AVG_ACCURACY_VS_ROUNDS_TITLE = 'Server Level Average Accuracy Across Rounds'
    SERVER_LEVEL_AVG_ACCURACY_VS_ROUNDS_PNG = 'server_level_avg_accuracy_vs_rounds.png'
    SERVER_OVERALL_AVG_ACCURACY_VS_ROUNDS_TITLE = 'Server Overall Average Accuracy Across Rounds'
    SERVER_OVERALL_AVG_ACCURACY_VS_ROUNDS_PNG = 'server_overall_avg_accuracy_vs_rounds.png'


# Drift patterns
class DriftPatterns:
    ABRUPT = 'abrupt'  # Implemented
    GRADUAL = 'gradual'  # Implemented
    INCREMENTAL = 'incremental'  # Implemented
    # REOCCURRING = 'reoccurring'
    # INCREMENTAL_ABRUPT = 'incre-abrupt'
    # ABRUPT_REOCURRING = 'abrupt-reoc'
    GRADUAL_REOCCURRING = 'grad-reoc'  # Implemented
    # OUT_OF_CONTROL = 'out-of-control'


# Drift creation methods
class DriftCreationMethods:
    LABEL_SWAPPING = 'label_swapping'
    ROTATION = 'rotation'


# file names related to logging
class Logs:
    CLIENT_LOG = 'client_log'
    SERVER_LOG = 'server_log'
    NON_DRIFTED_CLIENT_LOG = 'non_drifted_client_log'
    DRIFTED_CLIENT_LOG = 'drifted_client_log'
    NON_DRIFTED_CLIENT_AVG_LOG = 'non_drifted_client_avg_log'
    DRIFTED_CLIENT_AVG_LOG = 'drifted_client_avg_log'
    SERVER_LVL_AVG_LOG = 'server_level_avg_log'
    SERVER_OVERALL_AVG_LOG = 'server_overall_avg_log'


class ModelSettings:
    SAVE_MODEL_STATE = False  # Whether to save the model state after training
    LOAD_MODEL_STATE = True  # Whether to load the model state before training
