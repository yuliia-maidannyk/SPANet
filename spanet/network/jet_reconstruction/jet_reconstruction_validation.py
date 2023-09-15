from typing import Dict, Callable
import warnings

import numpy as np
import torch

from sklearn import metrics as sk_metrics

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.jet_reconstruction.jet_reconstruction_training import JetReconstructionTraining


class JetReconstructionValidation(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(options, torch_script)
        print("\nNow in validation.__init__...")
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)

    @property
    def particle_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            "accuracy": sk_metrics.accuracy_score,
            "sensitivity": sk_metrics.recall_score,
            "specificity": lambda t, p: sk_metrics.recall_score(~t, ~p),
            "f_score": sk_metrics.f1_score
        }

    @property
    def particle_score_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            # "roc_auc": sk_metrics.roc_auc_score,
            # "average_precision": sk_metrics.average_precision_score
        }

    def compute_metrics(self, jet_predictions, particle_scores, stacked_targets, stacked_masks):
        print("\nNow in validation.compute_metrics()...")
        event_permutation_group = self.event_permutation_tensor.cpu().numpy()
        num_permutations = len(event_permutation_group)
        num_targets, batch_size = stacked_masks.shape
        particle_predictions = particle_scores >= 0.5

        # Compute all possible target permutations and take the best performing permutation
        # First compute raw_old accuracy so that we can get an accuracy score for each event
        # This will also act as the method for choosing the best permutation to compare for the other metrics.
        jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        particle_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        for i, permutation in enumerate(event_permutation_group):
            for j, (prediction, target) in enumerate(zip(jet_predictions, stacked_targets[permutation])):
                jet_accuracies[i, j] = np.all(prediction == target, axis=1)

            particle_accuracies[i] = stacked_masks[permutation] == particle_predictions

        jet_accuracies = jet_accuracies.sum(1)
        particle_accuracies = particle_accuracies.sum(1)

        # Select the primary permutation which we will use for all other metrics.
        chosen_permutations = self.event_permutation_tensor[jet_accuracies.argmax(0)].T
        chosen_permutations = chosen_permutations.cpu()
        permuted_masks = torch.gather(torch.from_numpy(stacked_masks), 0, chosen_permutations).numpy()

        # Compute final accuracy vectors for output
        num_particles = stacked_masks.sum(0)
        jet_accuracies = jet_accuracies.max(0)
        particle_accuracies = particle_accuracies.max(0)

        # Create the logging dictionaries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
    
            metrics = {f"jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                    for j in range(1, num_targets + 1)
                    for i in range(1, j + 1)}

            metrics.update({f"particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                            for j in range(1, num_targets + 1)
                            for i in range(1, j + 1)})

        particle_scores = particle_scores.ravel()
        particle_targets = permuted_masks.ravel()
        particle_predictions = particle_predictions.ravel()

        for name, metric in self.particle_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_predictions)

        for name, metric in self.particle_score_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_scores)

        # Compute the sum accuracy of all complete events to act as our target for
        # early stopping, hyperparameter optimization, learning rate scheduling, etc.
        metrics["validation_accuracy"] = metrics[f"jet/accuracy_{num_targets}_of_{num_targets}"]
        #print(metrics)
        return metrics

    def validation_step(self, batch, batch_idx) -> Dict[str, np.float32]:
        print("\nNow in validation.validation_step()...")
        # Run the base prediction step
        sources, num_jets, targets, regression_targets, classification_targets = batch
        jet_predictions, particle_scores, regressions, classifications = self.predict(sources)

        # jet_predictions is assignments and particle_scores is detections

        batch_size = num_jets.shape[0]
        num_targets = len(targets)

        # Stack all of the targets into single array, we will also move to numpy for easier the numba computations.
        stacked_targets = np.zeros(num_targets, dtype=object)
        stacked_masks = np.zeros((num_targets, batch_size), dtype=bool)
        for i, (target, mask) in enumerate(targets):
            stacked_targets[i] = target.detach().cpu().numpy()
            stacked_masks[i] = mask.detach().cpu().numpy()

        regression_targets = {
            key: value.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }

        classification_targets = {
            key: value.detach().cpu().numpy()
            for key, value in classification_targets.items()
        }

        metrics = self.evaluator.full_report_string(jet_predictions, stacked_targets, stacked_masks, prefix="Purity/")

        # Apply permutation groups for each target
        #print("\nApplying permutation groups for each target...")
        for target, prediction, decoder in zip(stacked_targets, jet_predictions, self.branch_decoders):
            #print("\ntarget, prediction: ", target, prediction)
            for indices in decoder.permutation_indices:
                #print("\nindices in decoder.permutation_indices: ", indices)
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

        metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks))

        for key in regressions:
            delta = regressions[key] - regression_targets[key]
            
            percent_error = np.abs(delta / regression_targets[key])
            self.log(f"REGRESSION/{key}_percent_error", percent_error.mean(), sync_dist=True)

            absolute_error = np.abs(delta)
            self.log(f"REGRESSION/{key}_absolute_error", absolute_error.mean(), sync_dist=True)

            percent_deviation = delta / regression_targets[key]
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_percent_deviation", percent_deviation, self.global_step)

            absolute_deviation = delta
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_absolute_deviation", absolute_deviation, self.global_step)

        for key in classifications:
            accuracy = (classifications[key] == classification_targets[key])
            self.log(f"CLASSIFICATION/{key}_accuracy", accuracy.mean(), sync_dist=True)

        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(name, value, sync_dist=True)

        outputs = self.forward(batch.sources) # jet_reconstruction_network

        symmetric_losses, best_indices = self.symmetric_losses(
            outputs.assignments,
            outputs.detections,
            batch.assignment_targets
        )

        # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target.mask for target in batch.assignment_targets])
        masks = torch.gather(masks, 0, permutations)

        # Default unity weight on correct device.
        weights = torch.ones_like(symmetric_losses)

        # Balance based on the particles present - only used in partial event training
        if self.balance_particles:
            class_indices = (masks * self.particle_index_tensor.unsqueeze(1)).sum(0)
            weights *= self.particle_weights_tensor[class_indices]

        # Balance based on the number of jets in this event
        if self.balance_jets:
            weights *= self.jet_weights_tensor[batch.num_vectors]

        # Take the weighted average of the symmetric loss terms.
        masks = masks.unsqueeze(1)
        symmetric_losses = (weights * symmetric_losses).sum(-1) / masks.sum(-1)
        assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)

        total_loss = []

        if self.options.assignment_loss_scale > 0:
            total_loss.append(assignment_loss)

        if self.options.detection_loss_scale > 0:
            total_loss.append(detection_loss)
        
        total_loss = torch.cat([loss.view(-1) for loss in total_loss])

        self.log("validation_loss/validation_loss", total_loss.sum(), sync_dist=True)
        
        return metrics

<<<<<<< HEAD
    def validation_epoch_end(self, outputs):
        # Optionally use this accuracy score for something like hyperparameter search
        validation_accuracy = sum(x['validation_accuracy'] for x in outputs) / len(outputs)

        if self.options.verbose_output:
            for name, parameter in self.named_parameters():
                self.logger.experiment.add_histogram(name, parameter)

=======
>>>>>>> d8ee65ea8741689837c6e6c4e6aaffda8898fa08
    def test_step(self, batch, batch_idx):
        print("\nNow in validation.test_step()...")
        return self.validation_step(batch, batch_idx)