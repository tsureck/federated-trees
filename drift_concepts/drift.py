"""
Description: This module defines the attributes of the concept drift.

Author: Nisal Hemadasa
Date: 29-10-2024
Version: 1.0
"""
import bisect
import copy
import math
from typing import Dict, List
import os
from matplotlib.pyplot import imsave

import numpy as np
import torch
from scipy.ndimage import rotate
import torchvision.transforms as transforms

import constants
from federated_network.client import Client

def smooth_ramp(x: float, x0: float, x1: float, L: float = 6.0) -> float:
    """0 until x0, smooth logistic ramp to 1 by x1, then stay at 1."""
    if x <= x0:
        return 0.0
    if x >= x1:
        return 1.0

    # Map x from [x0, x1] -> [-L, +L]
    t = (x - x0) / (x1 - x0)          # in (0, 1)
    z = 2.0 * L * (t - 0.5)           # in (-L, +L)

    return 1.0 / (1.0 + math.exp(-z))


class Drift:
    def __init__(self, num_drifted_clients, drift_localization_factor, is_synchronous, async_drift_specs, drift_pattern,
                 drift_method, drift_start_round, drift_end_round, drifted_client_indices, max_rotation,
                 class_pairs_to_swap, num_drift_cycles=1, swap_direction='unidirectional', classes_to_rotate=None):
        # Number of clients to be applied with drifted data
        self.num_drifted_clients = num_drifted_clients

        # Factor to localize the drift to a certain concentrated group of clients. The value ranges from 0 to 1. E.g.,
        # 0.25 indicates that all drifted clients are concentrated to the first 0.25 indices of the clients.
        self.drift_localization_factor = drift_localization_factor

        # If the drift is synchronous or asynchronous
        self.is_synchronous = is_synchronous

        # Drift specifications for asynchronous drift
        self.async_drift_specs = async_drift_specs

        # Drift pattern, i.e., abrupt, gradual, incremental, reoccurring, incr-abrupt-reoc, incr-reoc, out-of-control.
        self.drift_pattern = drift_pattern

        # Label-swapping, rotations
        self.drift_method = drift_method

        # Defines the period in training rounds which drift starts appearing in clients
        self.drift_start_round = drift_start_round
        self.drift_end_round = drift_end_round

        # List of clients that have drifted data
        self.drifted_client_indices = drifted_client_indices
        print("Drifted clients: ", self.drifted_client_indices)

        # Maximum rotation angle for the drift created by rotations
        self.max_rotation = max_rotation

        # Classes to be swapped in the label-swapping drift method
        self.class_pairs_to_swap = class_pairs_to_swap

        # Current round of the federated training
        self.current_round = 0

        # Flag ot indicate when the drift is being applied to the client data
        self.is_drift = False

        # Flag to indicate if the drift is applied on the client data at least once before
        self.is_already_applied = False

        # Logging the drift transition
        self._applied_drift_logging = []

        # # Applied rotation angle (for rotation drift method)
        # self.applied_drift = 0

        # Number of drift cycles
        self.num_drift_cycles = num_drift_cycles

        # Direction of label swapping: 'bidirectional' (swap A<->B) or 'unidirectional' (only A to B)
        self.swap_direction = swap_direction

        # Classes to rotate: if empty, rotate all images; else, rotate only images of these classes
        self.classes_to_rotate = classes_to_rotate if classes_to_rotate is not None else []

        # Track original class counts for accurate cumulative swapping
        self.original_class_counts = {}

    def rotate_images(self, clients: List[Client], plot_save_path: str) -> List[Client]:
        """
        Apply rotation drift to the images of the client dataset. Both the rotation angle and the number of images to
        rotate increase linearly with the number of federated training rounds.
        :param clients: List of Client objects
        :param plot_save_path: Path to save plots
        :return: List of Client objects with the rotated images in their datasets
        """

        def apply_rotation(dataset, angle):
            """
            Apply rotation drift to a fraction of the images.
            :param dataset: Dataset to process
            :param _rotation_angle: Angle of rotation
            :return: Drifted images and original labels
            """
            _images = dataset.data  # Access dataset images
            _labels = dataset.targets  # Access dataset labels
            _drifted_images = _images.clone()

            for i in range(len(_images)):
                if not self.classes_to_rotate or _labels[i].item() in self.classes_to_rotate:
                    rotated_image = rotate(_images[i].numpy(), angle, reshape=False)
                    _drifted_images[i] = torch.tensor(rotated_image)

            return _drifted_images, _labels

        transition_progress = 0.0

        # Calculate rotation parameters
        match self.drift_pattern:
            case constants.DriftPatterns.INCREMENTAL:
                transition_progress = ((self.current_round + 1) - self.drift_start_round) / (
                        self.drift_end_round - self.drift_start_round)
                transition_progress = min(max(transition_progress, 0.0), 1.0)
            case constants.DriftPatterns.GRADUAL:
                transition_progress = smooth_ramp(self.current_round, self.drift_start_round, self.drift_end_round)
            case constants.DriftPatterns.ABRUPT:
                if self.current_round >= self.drift_start_round:
                    transition_progress = 1.0
                else:
                    transition_progress = 0.0
            case constants.DriftPatterns.GRADUAL_REOCCURRING:
                if self.current_round < self.drift_start_round or self.current_round > self.drift_end_round:
                    transition_progress = 0.0
                else:
                    cycle_length = self.drift_end_round - self.drift_start_round + 1
                    position_in_cycle = (self.current_round - self.drift_start_round) % cycle_length
                    # Use num_drift_cycles to have multiple cycles across the drift period
                    transition_progress = (np.sin(
                        2 * self.num_drift_cycles * (position_in_cycle + 1) / cycle_length * np.pi - np.pi/2
                    ) + 1) / 2.0

        rotation_angle = transition_progress * self.max_rotation
        self._applied_drift_logging.append(rotation_angle)

        if self.drift_start_round >= self.drift_end_round:
            return clients

        first_drifted_client = None
        # Check if there are drifted clients
        if self.drifted_client_indices:
            # Identify the first drifted client to process the dataset and duplicate a copy (not the reference)
            first_drifted_client = copy.deepcopy(clients[self.drifted_client_indices[0]])

            # Process training dataset
            train_images, train_labels = apply_rotation(first_drifted_client.local_trainset.dataset, rotation_angle)
            first_drifted_client.local_trainset.dataset.data = train_images
            first_drifted_client.local_trainset.dataset.targets = train_labels

            # Process testing dataset
            test_images, test_labels = apply_rotation(first_drifted_client.testset.dataset, rotation_angle)
            first_drifted_client.testset.dataset.data = test_images
            first_drifted_client.testset.dataset.targets = test_labels

            # Assign the updated datasets to all drifted clients, since they share the same data
            for idx in self.drifted_client_indices:
                clients[idx].local_trainset.dataset = first_drifted_client.local_trainset.dataset
                clients[idx].testset.dataset = first_drifted_client.testset.dataset

        if first_drifted_client:
            if first_drifted_client.local_trainset:
                os.makedirs(plot_save_path, exist_ok=True)
                imsave(f"{plot_save_path}image_in_round_{self.current_round}.png", first_drifted_client.local_trainset.dataset.data[0].numpy())
        return clients

    def swap_labels(self, clients: List[Client]) -> List[Client]:
        """
        Swap the labels of the specified classes in the training and testing sets for drifted clients.
        :param clients: List of Client objects
        :return: Updated list of Client objects with swapped labels in their datasets
        """

        def swap_labels_in_dataset(dataset, transition_progress):
            """
            Swap labels in a dataset based on the class pairs to swap, ensuring cumulative fraction.
            :param dataset: Dataset to process
            :param transition_progress: Current transition progress (0 to 1)
            :return: Updated images and labels tensors
            """
            images = dataset.data  # Access dataset images
            labels = dataset.targets  # Access dataset labels

            for class_a, class_b in self.class_pairs_to_swap:
                if self.swap_direction == 'unidirectional':
                    indices_a = (labels == class_a).nonzero(as_tuple=True)[0]

                    # Calculate target number to swap cumulatively
                    target_swapped = int(transition_progress * len(indices_a))

                    # Randomly select indices to swap
                    indices_a_to_swap = indices_a[torch.randperm(len(indices_a))[:target_swapped]]

                    # Change labels from class_a to class_b
                    labels[indices_a_to_swap] = class_b

                if self.swap_direction == 'bidirectional':
                    indices_a = (labels == class_a).nonzero(as_tuple=True)[0]
                    indices_b = (labels == class_b).nonzero(as_tuple=True)[0]

                    # Calculate target number to swap cumulatively
                    target_swapped = int(transition_progress * min(len(indices_a), len(indices_b)))

                    # Randomly select indices to swap from class_a to class_b
                    indices_a_to_swap = indices_a[torch.randperm(len(indices_a))[:target_swapped]]
                    labels[indices_a_to_swap] = class_b

                    # Randomly select indices to swap from class_b to class_a
                    indices_b_to_swap = indices_b[torch.randperm(len(indices_b))[:target_swapped]]
                    labels[indices_b_to_swap] = class_a

            return images, labels

        transition_progress = 0.0

        # Calculate transition progress based on drift pattern
        match self.drift_pattern:
            case constants.DriftPatterns.INCREMENTAL:
                transition_progress = ((self.current_round + 1) - self.drift_start_round) / (
                        self.drift_end_round - self.drift_start_round)
                transition_progress = min(max(transition_progress, 0.0), 1.0)
            case constants.DriftPatterns.GRADUAL:
                transition_progress = smooth_ramp(self.current_round, self.drift_start_round, self.drift_end_round)
            case constants.DriftPatterns.ABRUPT:
                if self.current_round >= self.drift_start_round:
                    transition_progress = 1.0
                else:
                    transition_progress = 0.0
            case constants.DriftPatterns.GRADUAL_REOCCURRING:
                if self.current_round < self.drift_start_round or self.current_round >= self.drift_end_round:
                    transition_progress = 0.0
                else:
                    cycle_length = self.drift_end_round - self.drift_start_round + 1
                    position_in_cycle = (self.current_round - self.drift_start_round) % cycle_length
                    # Use num_drift_cycles to have multiple cycles across the drift period
                    transition_progress = (np.sin(
                        2 * self.num_drift_cycles * (position_in_cycle + 1) / cycle_length * np.pi - np.pi/2
                    ) + 1) / 2.0

                    print(cycle_length)
                    print(position_in_cycle)
                    print(transition_progress)

        # Log the transition progress
        self._applied_drift_logging.append(transition_progress)

        # Check if there are drifted clients
        if self.drifted_client_indices:
            # Identify the first drifted client to process the dataset and duplicate a copy (not the reference)
            first_drifted_client = copy.deepcopy(clients[self.drifted_client_indices[0]])

            # Process training dataset
            train_images, train_labels = swap_labels_in_dataset(first_drifted_client.local_trainset.dataset, transition_progress)
            first_drifted_client.local_trainset.dataset.data = train_images
            first_drifted_client.local_trainset.dataset.targets = train_labels

            # Process testing dataset
            test_images, test_labels = swap_labels_in_dataset(first_drifted_client.testset.dataset, transition_progress)
            first_drifted_client.testset.dataset.data = test_images
            first_drifted_client.testset.dataset.targets = test_labels

            # Assign the updated datasets to all drifted clients, since they share the same data
            for idx in self.drifted_client_indices:
                clients[idx].local_trainset.dataset = first_drifted_client.local_trainset.dataset
                clients[idx].testset.dataset = first_drifted_client.testset.dataset

        return clients

def get_clients_with_drift(_num_client_instances: int, _clients_fraction_with_drift: float,
                           drift_localization_factor: float, is_synchronous: bool, async_drift_specs: Dict) -> list:
    """
    Get the list of clients that have drifted data.
    :param _num_client_instances: Total number of client instances in the federated network
    :param _clients_fraction_with_drift: Fraction of clients with drifted data
    :param drift_localization_factor: Factor to localize the drift to a certain concentrated group of clients
    :param is_synchronous: Boolean indicating if the drift is synchronous or asynchronous
    :param async_drift_specs: Dictionary containing the specifications for asynchronous drift
    :return: Indices of clients with drifted data
    """
    num_clients_with_drift = int(_clients_fraction_with_drift * _num_client_instances * drift_localization_factor)
    # The cohort of clients with the possibility of drift occurrence
    _num_client_cohort_with_drift = int(_num_client_instances * drift_localization_factor)
    client_indices = torch.randperm(_num_client_cohort_with_drift).tolist()
    drift_clients = client_indices[:num_clients_with_drift]

    if not is_synchronous:
        drifted_client_grouping_margin_indices = [
            math.ceil(i * (_num_client_instances / async_drift_specs['num_drift_groups']))
            for i in range(1, async_drift_specs['num_drift_groups'])
        ]

        # Initialize an empty list to store the grouped lists
        grouped_drifted_clients = []
        # Sort the drifted client indices to ensure proper ordering
        sorted_drift_clients = sorted(drift_clients)
        # Iterate over the drifted_client_grouping_margin_indices and progressively build the groups
        current_group = []

        # Each element in 'drift_groups' is a cumulative list of all previous groups, plus the new values up to the
        # current margin
        for margin in drifted_client_grouping_margin_indices:
            # Add all client indices that are less than or equal to the current margin
            current_group.extend(idx for idx in sorted_drift_clients if idx < margin and idx not in current_group)
            # Append a copy of the current_group to avoid modifying previous lists
            grouped_drifted_clients.append(current_group[:])
        # Add the final group containing all indices
        grouped_drifted_clients.append(sorted_drift_clients)

        # For testing purposes
        grouped_drifted_clients = [list(set(grouped_drifted_clients[1]) - set(grouped_drifted_clients[0])),
                                   grouped_drifted_clients[1]]

        async_drift_specs['drift_groups'] = grouped_drifted_clients

    return drift_clients


def modify_drifted_client_groups(drift: Drift, _round: int) -> None:
    """"
    Modify the drifted client groups based on the drift specifications.
    :param drift: Drift object
    :param _round: Current training round
    :return: None
    """
    # TODO: This function is only desinged for 2 groups. Generalize for more than 2 groups
    if _round < drift.async_drift_specs['drift_split_round']:
        drift.drifted_client_indices = drift.async_drift_specs['drift_groups'][0]
    else:
        # Leave the drift from one group and add another drift to the second group?
        if not drift.drifted_client_indices == drift.async_drift_specs['drift_groups'][1]:
            drift.is_already_applied = False
            drift.drifted_client_indices = drift.async_drift_specs['drift_groups'][1]


def drift_fn(num_client_instances: int, num_training_rounds: int, drift_specs: Dict) -> Drift:
    """
    Create a drift object using the specifications given as inputs.
    :param num_client_instances: Total number of client instances in the federated network
    :param num_training_rounds: Total number of training rounds
    :param drift_specs: Dictionary containing the drift specifications
    :return: Drift object
    """
    # Drift start and end rounds
    drift_start_round = math.ceil(drift_specs['drift_start_round'] * num_training_rounds)
    drift_end_round = math.ceil(drift_specs['drift_end_round'] * num_training_rounds)
    print("Drift start round: ", drift_start_round)
    print("Drift end round: ", drift_end_round)

    # The round which the drift starts to affect on the second group of drifted clients in asynchronous case
    if not drift_specs['is_synchronous']:
        drift_duration = drift_end_round - drift_start_round  # Duration of the drift in rounds
        drift_split_round = math.ceil(
            drift_start_round + drift_specs['async_drift_specs']['drift_split_round'] * drift_duration)
        print("Async drift split round: ", drift_split_round)
        drift_specs['async_drift_specs']['drift_split_round'] = drift_split_round

    return Drift(num_drifted_clients=drift_specs['clients_fraction'] * num_client_instances,
                 drift_localization_factor=drift_specs['drift_localization_factor'],
                 is_synchronous=drift_specs['is_synchronous'],
                 async_drift_specs=drift_specs['async_drift_specs'],
                 drift_pattern=drift_specs['drift_pattern'],
                 drift_method=drift_specs['drift_method'],
                 drift_start_round=math.ceil(drift_specs['drift_start_round'] * num_training_rounds),
                 drift_end_round=math.ceil(drift_specs['drift_end_round'] * num_training_rounds),
                 drifted_client_indices=get_clients_with_drift(num_client_instances, drift_specs['clients_fraction'],
                                                               drift_specs['drift_localization_factor'],
                                                               drift_specs['is_synchronous'],
                                                               drift_specs['async_drift_specs']),
                 max_rotation=drift_specs['max_rotation'],
                 class_pairs_to_swap=drift_specs['class_pairs_to_swap'],
                 num_drift_cycles=drift_specs['num_drift_cycles'],
                 swap_direction=drift_specs.get('swap_direction', 'bidirectional'),
                 classes_to_rotate=drift_specs.get('classes_to_rotate', []))


def apply_drift(clients: List[Client], drift: Drift, plot_save_path: str) -> List[Client]:
    """
    Apply drift to the training data of the clients.
    :param clients: List of Client objects
    :param drift: Drift object
    :return: List of Client objects with drifted data (dataloaders)
    """
    for client in clients:
        client.sample_data()

    match drift.drift_method:
        case constants.DriftCreationMethods.LABEL_SWAPPING:
            return drift.swap_labels(clients)
        case constants.DriftCreationMethods.ROTATION:
            return drift.rotate_images(clients, plot_save_path)
        case _:
            print("Drift method not recognized. No drift applied.")

    return clients