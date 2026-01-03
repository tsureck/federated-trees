"""
Description: This file consists the Averaging functions for FedAvg algorithm.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""

import torch


class FedAvg:
    def __init__(self):
        pass

    # TODO: Adjust here for different aggregation strategies or detecting malicious clients
    def aggregate_models(self, model, client_model_params_list):
        """ Aggregate the client models to the global model and returns the new aggregated model"""
        model_params = model.state_dict()
        for i in model_params.keys():
            model_params[i] = torch.stack(
                [client_model_params[i].float() for client_model_params in client_model_params_list], 0).mean(0)

        model.load_state_dict(model_params)
        return model


def aggregator_fn():
    """ Returns an instance of the FedAvg aggregation strategy """
    _strategy = FedAvg()
    return _strategy

    # def aggregate_evaluate(self, rnd, results, failures):
    #     if not results:
    #         return None, {}
    #
    #     # Aggregate loss
    #     loss = sum([r[1].num_examples * r[1].loss for r in results]) / sum(
    #         [r[1].num_examples for r in results])
    #
    #     # Aggregate accuracy
    #     accuracy = sum([r[1].num_examples * r[1].metrics["accuracy"] for r in results]) / sum(
    #         [r[1].num_examples for r in results])
    #
    #     print(f"Round {rnd} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    #     return loss, {"accuracy": accuracy}

#
#
# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     """
#     Aggregate the metrics using a weighted average.
#     :param metrics: weights of the client models to be aggregated.
#     :return: weighted average of the metrics.
#     """
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
#
#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}
