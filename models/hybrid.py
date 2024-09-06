# Import necessary libraries
import xarray as xr  # For handling multi-dimensional data, useful in geospatial and climate datasets
import numpy as np  # For handling arrays and mathematical operations
import torch  # The core deep learning library
import torch.nn.functional as torchf  # Functional modules for operations like activation functions
import pytorch_lightning as pl  # PyTorch Lightning, a high-level library simplifying training
from typing import List, Dict, Tuple  # For type hinting

from utils.data_utils import Normalize  # Custom normalization class (assumed external)
from models.feedforward import FeedForward  # A feedforward neural network (assumed external)

# The Q10Model class, extending PyTorch Lightning's LightningModule
class Q10Model(pl.LightningModule):
    """
    Q10Model is a hybrid model combining a neural network (NN) and a physical model
    to predict temperature-dependent processes (e.g., respiration).
    """

    def __init__(
            self,
            features: List[str],  # Input feature names (e.g., temperature, humidity)
            targets: List[str],  # Target variable names (e.g., respiration rate)
            norm: Normalize,  # Normalization instance to scale features and targets
            ds: xr.Dataset,  # Dataset used for storing results, likely in xarray format
            q10_init: float = 1.5,  # Initial value of the Q10 parameter (1.5 is typical for biological rates)
            hidden_dim: int = 128,  # Number of neurons in the hidden layer of the neural network
            num_layers: int = 2,  # Number of layers in the feedforward neural network
            learning_rate: float = 0.01,  # Learning rate for training
            weight_decay: float = 0.1,  # Weight decay (regularization) to prevent overfitting
            dropout: float = 0.,  # Dropout rate to prevent overfitting (0 means no dropout)
            activation: str = 'tanh',  # Activation function used in the NN (can be 'tanh', 'relu', etc.)
            num_steps: int = 0  # Track how many steps the model has trained for
    ) -> None:
        """
        Initialize the Q10Model with parameters for the neural network, physical model, and optimization settings.
        """

        # Call the parent class's initializer
        super().__init__()

        # Save hyperparameters (this allows Lightning to store and manage these parameters automatically)
        self.save_hyperparameters(
            'features',
            'targets',
            'q10_init',
            'hidden_dim',
            'num_layers',
            'dropout',
            'activation',
            'learning_rate',
            'weight_decay'
        )

        # Initialize the features and targets, which are lists of variable names
        self.features = features  # Input feature names
        self.targets = targets  # Target (output) variable names

        # Initialize the Q10 parameter, which is trainable and controls temperature dependence
        self.q10_init = q10_init
        self.q10 = torch.nn.Parameter(torch.ones(1) * self.q10_init)  # Make Q10 a trainable parameter

        # Set the reference temperature, typically 15°C, used in the Q10 formula
        self.ta_ref = 15.0

        # Initialize a normalization layer for the input features (e.g., scaling the input data)
        self.input_norm = norm.get_normalization_layer(variables=self.features, invert=False, stack=True)

        # Initialize the neural network (FeedForward is an assumed external model)
        self.nn = FeedForward(
            num_inputs=len(self.features),  # Number of input features
            num_outputs=len(self.targets),  # Number of output variables
            num_hidden=hidden_dim,  # Number of neurons in hidden layers
            num_layers=num_layers,  # Number of layers in the feedforward network
            dropout=dropout,  # Dropout rate to prevent overfitting
            dropout_last=False,  # Apply dropout to the last layer
            activation=activation,  # The activation function to use (e.g., 'tanh')
        )

        # Initialize normalization layers for the targets (output variables)
        self.target_norm = norm.get_normalization_layer(variables=self.targets, invert=False, stack=True)
        self.target_denorm = norm.get_normalization_layer(variables=self.targets, invert=True, stack=True)

        # Initialize the loss function (MSE or Mean Squared Error)
        self.criterion = torch.nn.MSELoss()

        # Store the dataset where we will save the model's predictions and other results
        self.ds = ds

        # Keep track of how many training steps the model has undergone
        self.num_steps = num_steps

        # Pre-allocate space to store the history of the Q10 parameter during training
        self.q10_history = np.zeros(100000, dtype=np.float32) * np.nan

    # Forward method: defines how data flows through the model
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        - Takes input data `x`, normalizes it, passes it through the neural network,
          and applies the Q10 equation for temperature adjustment.
        """

        # Normalize the input data using the input normalization layer
        z = self.input_norm(x)

        # Pass the normalized input data through the neural network
        z = self.nn(z)

        # Use softplus activation (a smooth version of ReLU) on the network output
        rb = torchf.softplus(z)

        # Apply the Q10 equation to model temperature dependence
        # `reco` is the predicted respiration rate after applying the Q10 formula
        reco = rb * self.q10 ** (0.1 * (x['ta'] - self.ta_ref))

        # Return both `reco` (the final prediction) and `rb` (intermediate NN output)
        return reco, rb

    # Loss function: calculates loss on normalized data
    def criterion_normed(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss (MSE) on normalized predictions and targets.
        - `y_hat`: Predicted values (normalized).
        - `y`: Actual target values (normalized).
        """
        return self.criterion(self.target_norm(y_hat), self.target_norm(y))

    # Defines what happens during a training step (one batch of data)
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step: processes one batch of data, calculates loss, and logs the Q10 parameter.
        - `batch`: The input data and targets.
        - `batch_idx`: Index of the batch (useful for logging).
        """
        # Unpack the batch: the data and the time index (time not used here)
        batch, _ = batch

        # Perform a forward pass to get the model's predictions
        reco_hat, _ = self(batch)

        # Calculate the loss on the predictions and actual target data
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Save the current Q10 parameter to track its evolution during training
        self.q10_history[self.global_step] = self.q10.item()

        # Log the training loss and the current Q10 parameter (to monitor during training)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('q10', self.q10, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        # Return the loss (PyTorch Lightning will handle the backpropagation automatically)
        return loss

    # Defines what happens during a validation step (like training, but without backprop)
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step: processes a batch of validation data and logs predictions.
        - `batch`: The input data and targets.
        - `batch_idx`: Index of the batch (useful for logging).
        """
        # Unpack the batch: data and the index (index is used for storing predictions)
        batch, idx = batch

        # Perform a forward pass to get predictions
        reco_hat, rb_hat = self(batch)

        # Calculate the loss on validation data
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Log the validation loss
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Return predictions and index (useful for storing the results)
        return {'reco_hat': reco_hat, 'rb_hat': rb_hat, 'idx': idx}

    # Combine validation results at the end of each validation epoch
    def validation_epoch_end(self, validation_step_outputs) -> None:
        """
        At the end of each validation epoch, store the predictions in the dataset.
        - `validation_step_outputs`: A list of results from each validation step.
        """
        for item in validation_step_outputs:
            reco_hat = item['reco_hat'][:, 0].cpu()  # Get predicted `reco` values
            rb_hat = item['rb_hat'][:, 0].cpu()  # Get intermediate `rb` values
            idx = item['idx'].cpu()  # Get the time index

            # Store the predictions in the dataset at the corresponding time steps
            self.ds['reco_pred'].values[self.current_epoch, idx] = reco_hat
            self.ds['rb_pred'].values[self.current_epoch, idx] = rb_hat

    # Define what happens during a test step (similar to validation, but for testing data)
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step: processes a batch of test data and calculates the loss.
        - `batch`: The input data and targets.
        - `batch_idx`: Index of the batch (useful for logging).
        """
        # Unpack the batch
        batch, _ = batch

        # Perform a forward pass to get predictions
        reco_hat, _ = self(batch)

        # Calculate the loss on test data
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Log the test loss
        self.log('test_loss', loss)

        return {'test_loss': loss}

    # Configuring the optimizer for backpropagation (AdamW optimizer is used here)
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Set up the optimizer that will be used during training.
        We are using AdamW, a variant of Adam that includes weight decay.
        """
        # Create an AdamW optimizer, which adjusts the neural network weights and Q10 parameter
        optimizer = torch.optim.AdamW(
            [
                {
                    # The neural network parameters are optimized with regular weight decay and learning rate
                    'params': self.nn.parameters(),
                    'weight_decay': self.hparams.weight_decay,
                    'learning_rate': self.hparams.learning_rate
                },
                {
                    # The Q10 parameter is optimized separately with no weight decay and a higher learning rate
                    'params': [self.q10],
                    'weight_decay': 0.0,
                    'learning_rate': self.hparams.learning_rate * 10
                }
            ]
        )

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Helper function to add model-specific arguments to the argument parser.
        This is used when running the model with a command-line interface (CLI).
        """
        # Extend the parent parser with additional arguments specific to Q10Model
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        # Add arguments related to the model architecture and training parameters
        parser.add_argument('--hidden_dim', type=int, default=16)  # Number of neurons in hidden layers
        parser.add_argument('--num_layers', type=int, default=2)  # Number of hidden layers
        parser.add_argument('--c', type=float, default=0.1)  # Some hyperparameter (perhaps unused here)
        parser.add_argument('--learning_rate', type=float, default=0.05)  # Learning rate
        parser.add_argument('--weight_decay', type=float, default=0.1)  # Weight decay for regularization

        # Return the updated parser
        return parser
