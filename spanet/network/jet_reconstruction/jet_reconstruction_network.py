import numpy as np
import torch
from torch import nn

from spanet.options import Options
from spanet.dataset.types import Tuple, Outputs, Source, Predictions

from spanet.network.layers.vector_encoder import JetEncoder
from spanet.network.layers.branch_decoder import BranchDecoder
from spanet.network.layers.embedding import MultiInputVectorEmbedding
from spanet.network.layers.regression_decoder import RegressionDecoder
from spanet.network.layers.classification_decoder import ClassificationDecoder

from spanet.network.prediction_selection import extract_predictions
from spanet.network.jet_reconstruction.jet_reconstruction_base import JetReconstructionBase

TArray = np.ndarray


class JetReconstructionNetwork(JetReconstructionBase):
    def __init__(self, options: Options, torch_script: bool = False):
        """ Base class defining the SPANet architecture.
        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """
        super(JetReconstructionNetwork, self).__init__(options)

        compile_module = torch.jit.script if torch_script else lambda x: x
        #print("\nNow in network.init...\n")
        self.hidden_dim = options.hidden_dim

        # (1) independent jet embeddings to produce latent space representations for each jet
        self.embedding = compile_module(MultiInputVectorEmbedding(
            options,
            self.training_dataset
        ))

        # (2) central stack of transformer encoders into fixed-length states
        self.encoder = compile_module(JetEncoder(
            options,
        ))

        # (3) braches of decoders back into a variable-length output
        self.branch_decoders = nn.ModuleList([
            BranchDecoder(
                options,
                event_particle_name,
                self.event_info.product_particles[event_particle_name].names,
                product_symmetry,
                self.enable_softmax
            )
            for event_particle_name, product_symmetry
            in self.event_info.product_symmetries.items()
        ])

        self.regression_decoder = compile_module(RegressionDecoder(
            options,
            self.training_dataset
        ))

        self.classification_decoder = compile_module(ClassificationDecoder(
            options,
            self.training_dataset
        ))

        # An example input for generating the network's graph, batch size of 2
        # self.example_input_array = tuple(x.contiguous() for x in self.training_dataset[:2][0])

    @property
    def enable_softmax(self):
        return True

    def forward(self, sources: Tuple[Source, ...]) -> Outputs:
        #print("\nNow in network.forward...\n")
        # Embed all of the different input regression_vectors into the same latent space.
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources)

        # Extract features from data using transformer
        #print("\nBegan transformer step...\n")
        hidden, event_vector = self.encoder(embeddings, padding_masks, sequence_masks)
        #print("\nFinished transformer step...\n")

        # Create output lists for each particle in event.
        assignments = []
        detections = []

        encoded_vectors = {
            "EVENT": event_vector
        }
        #print("encoded vectors shape: ", np.shape(encoded_vectors["EVENT"].numpy())) (1,64,32) -- (1, BATCH_SIZE, hidden_dim)

        # Pass the shared hidden state to every decoder branch
        #print("\nBegan decoder step...\n")
        for decoder in self.branch_decoders:
            (
                assignment,
                detection,
                assignment_mask,
                event_particle_vector,
                product_particle_vectors
            ) = decoder(hidden, padding_masks, sequence_masks, global_masks)

            assignments.append(assignment)
            detections.append(detection)

            # Assign the summarising vectors to their correct structure.
            encoded_vectors["/".join([decoder.particle_name, "PARTICLE"])] = event_particle_vector
            for product_name, product_vector in zip(decoder.product_names, product_particle_vectors):
                encoded_vectors["/".join([decoder.particle_name, product_name])] = product_vector

        # Predict the valid regressions for any real values associated with the event.
        regressions = self.regression_decoder(encoded_vectors)

        # Predict additional classification targets for any branch of the event.
        classifications = self.classification_decoder(encoded_vectors)
        #print("assignments len: ", len(assignments))
        #print("assignments[0] shape: ", np.shape(assignments[0]))
        #print("\nFinished decoder step...\n")
        #print(encoded_vectors)

        return Outputs(
            assignments,
            detections,
            encoded_vectors,
            regressions,
            classifications
        )

    def predict(self, sources: Tuple[Source, ...]) -> Predictions:
        #print("\nNow in network.predict...\n")
        with torch.no_grad():
            assignments, detections, _, regressions, classifications = self.forward(sources)
            #print("\nassignments[0] before extract_prediction(): ", np.shape(assignments[0]))
            #print("\nassignments[1] before extract_prediction(): ", np.shape(assignments[1]))
            #print("\nassignments[2] before extract_prediction(): ", np.shape(assignments[2]))
            # Assignments now have shapes [64,15,15,15], [64,15] and [64,15,15] for t1,t2,H

            # Extract assignment probabilities and find the least conflicting assignment. 
            # Gets called 64 times
            assignments = extract_predictions([
                np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
                for assignment in assignments
            ])

            # Assignments now have shapes [64,3], [64,1] and [64,2] for t1,t2,H

            #print("\nassignments[0] after extract_prediction(): ", np.shape(assignments[0]))
            #print("\nassignments[1] after extract_prediction(): ", np.shape(assignments[1]))
            #print("\nassignments[2] after extract_prediction(): ", np.shape(assignments[2]))   

            #print(assignments[0])

            # Convert detection logits into probabilities and move to CPU.
            detections = np.stack([
                torch.sigmoid(detection).cpu().numpy()
                for detection in detections
            ])

            # Move regressions to CPU and away from torch.
            regressions = {
                key: value.cpu().numpy()
                for key, value in regressions.items()
            }

            classifications = {
                key: value.cpu().argmax(1).numpy()
                for key, value in classifications.items()
            }

        return Predictions(
            assignments,
            detections,
            regressions,
            classifications
        )

    def predict_assignments(self, sources: Tuple[Source, ...]) -> np.ndarray:
        # Run the base prediction step
        with torch.no_grad():
            assignments = [
                np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
                for assignment in self.forward(sources)[0]
            ]

        # Find the optimal selection of jets from the output distributions.
        return extract_predictions(assignments)

    def predict_assignments_and_detections(self, sources: Tuple[Source, ...]) -> Tuple[TArray, TArray]:
        assignments, detections, regressions, classifications = self.predict(sources)

        # Always predict the particle exists if we didn't train on it
        if self.options.detection_loss_scale == 0:
            detections += 1

        return assignments, detections >= 0.5