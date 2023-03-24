import numpy as np
import tensorflow as tf

# from Study.Masks.Bias import maskBiases
# from Study.Masks.Weights import maskWeights

import logging

logger = logging.getLogger(__name__)


class LotteryTicketIterativePruningCallback(tf.keras.callbacks.Callback):

    def __init__(
        self,
        end_prune_epoch: int,
        pre_prune_epochs_k: int,
        j_steps: int,
        prune_amount: float,
    ):
        """
        This callback will prune the model at the startPruneEpoch epoch until the endEpoch epoch. Then it will reset the weights to the original weights with the mask applied at the end of training. 
        
        :param endEpoch: the epoch to end pruning
        :param percentagePerEpoch: the percentage to prune per epoch
        :param j_steps: number of pruning steps to take

        """
        super().__init__()

        # Check input
        assert j_steps > 0, "j_steps must be greater than 0"
        assert prune_amount > 0 and prune_amount < 1, "pruneAmount must be greater than 0 and less than 1"
        assert end_prune_epoch > 0, "endPruneEpoch must be greater than 0"
        assert j_steps < end_prune_epoch, "j_steps must be less than endPruneEpoch"

        # Configure initial state vars
        self.j_steps = j_steps
        self.pre_prune_epochs_k: int = pre_prune_epochs_k
        self.end_prune_epoch = end_prune_epoch
        self.pruneAmount = prune_amount
        self.prunePerEpoch = prune_amount / ((end_prune_epoch // j_steps) + 1)

        self.rewind_weights = None
        self.currentMask = None

        self.current_epoch = 0
        self.currentPercent = self.prunePerEpoch

        logger.debug(f"Prune Per Epoch: {self.prunePerEpoch}")

    ### SECTION: Keras callbacks

    def on_epoch_begin(self, epoch, logs=None):
        """
        Keras callback function, runs once at the start of each epoch.
        Most of the iterative pruning logic is implemented here.
        """
        logger.debug(
            f"on_epoch_begin: self.currentEpoch = {self.current_epoch}")
        logger.debug(
            f"on_epoch_begin: self.currentPercent = {self.currentPercent}")

        current_weights = self.get_trainableAndMaskedWeights()

        # Save the initial weights before the first epoch.
        if self.current_epoch == self.pre_prune_epochs_k:
            self.rewind_weights = current_weights

       

        # if self.currentEpoch == self.endPruneEpoch:
        #     self.iterateMask("updateWeights")

        self.current_epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        """
        Keras callback function, runs once at the end of each epoch.
        """
        
        # Epoch where we have to update the pruning masks
        if (self.current_epoch >= self.pre_prune_epochs_k) and 
            (self.current_epoch < self.end_prune_epoch) and (
                self.current_epoch % self.j_steps == 0):
            self.currentPercent = self.currentPercent + self.prunePerEpoch
            self.updateInternalMasks(
                current_weights
            )  # Sets self.mask to new masks based on self.currentPercent and currentWeights
            self.copyMasksToModel()

        self.assertions()

    ### SECTION: Helper Functions

    def _trainableMaskedLayers(self):
        """
        A list of layers which have trainable weights and kernel constraints
        """
        return [
            layer for layer in self.model.layers
            if len(layer.trainable_weights) > 0
            and getattr(layer, 'kernel_constraint', None) is not None
        ]

    def get_trainableAndMaskedWeights(self) -> list[np.array]:
        """
        Return the trainable weights of each layer as a list of numpy arrays
        """
        return [
            layer.trainable_weights[0].numpy()
            for layer in self._trainableMaskedLayers()
        ]

    def set_trainableAndMaskedWeights(self, new_weights: list) -> None:
        """
        Sets the trainable weights of each layer given a list of numpy arrays
        """
        assert len(self._trainableMaskedLayers()) == len(
            new_weights
        ), "Length of new weight list must equal number of layers in model."

        for layer, new_layer_weights in zip(self._trainableMaskedLayers(),
                                            new_weights):
            layer.trainable_weights[0].assign(new_layer_weights)

    def copyMasksToModel(self):
        """
        Updates the layer constraints in self.model to reflect the values in self.mask
        """
        maskedLayers = self._trainableMaskedLayers()

        assert len(self.mask) == len(
            maskedLayers
        ), "Length of new mask list must equal number of masked layers in model."

        for layer, newMask in zip(maskedLayers, self.mask):
            layer.kernel_constraint.updateMask(newMask)

    def updateInternalMasks(self, weights: list):
        """
        Creates new masks for weights, based on prune percentage, and stores this in self.mask
        """
        # flatten the mask
        flatWeights = np.concatenate([i.flatten() for i in weights])
        flatMask = np.ones(flatWeights.shape)

        # get the number of weights to prune
        numWeightsToPrune = int(self.currentPercent * len(flatWeights))

        # set the weights to prune to 0 where the absolute value of the weights is the smallest
        flatMask[np.where(
            np.argsort(np.abs(flatWeights)) < numWeightsToPrune)] = 0

        # reshape the mask
        newMask = []
        for i, layer in enumerate(weights):
            newMask.append(flatMask[:layer.size].reshape(layer.shape))
            flatMask = flatMask[layer.size:]

        self.mask = newMask

        logger.debug(
            f"updateInternalMask:numWeightsToPrune = {numWeightsToPrune}")

    def assertions(self):
        currentWeights = np.concatenate(
            [i.flatten() for i in self.get_trainableAndMaskedWeights()])
        currentMask = np.concatenate([i.flatten() for i in self.mask])

        logger.debug(
            f"assertions: number of zeros in mask = {(currentMask==0).sum()}")
        logger.debug(
            f"assertions: number of zeros in weights = {(currentWeights==0).sum()}"
        )

        assert ((currentMask == 0) & (currentWeights == 0) == (
            currentMask == 0
        )).all(
        ), "At least one nonzero weight was found which should have been masked."
        #assert ((currentMask==0) == (currentWeights==0)).all(), "Mask and weight zeros do not match"
