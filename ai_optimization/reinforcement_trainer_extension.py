"""
Reinforcement Learning Trainer Extensions

Extends the ReinforcementTrainer with additional methods for production use.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam

def create_baseline_model(self):
    """
    Create a simple baseline model when no trained model is available.
    This ensures the system can still provide predictions even without historical training.
    
    The baseline model uses basic heuristics and is biased toward standard execution
    for safety until proper training data is available.
    
    Returns:
        True on success
    """
    logging.info("Creating baseline model for initial predictions")
    
    try:
        # Create a simple model architecture
        inputs = Input(shape=(self.input_dim,))
        x = Dense(32, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(self.action_size, activation='linear')(x)
        
        # Create and compile the model
        simple_model = Model(inputs, outputs)
        simple_model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        
        # Initialize with a standard execution bias
        # We'll use a simple approach to make the model prefer standard execution (action 0)
        # until it's properly trained
        weights = simple_model.get_weights()
        
        # Set the bias in the final layer to favor standard execution
        # Last layer weights shape is typically (previous_layer_neurons, action_size)
        # Last layer bias shape is (action_size,) in this case [2]
        if len(weights) >= 2:
            output_bias = weights[-1]
            # Make standard execution (index 0) slightly preferred by default
            output_bias[0] = 0.2  # Positive bias for standard execution
            output_bias[1] = -0.1  # Slightly negative bias for flashloan execution
            weights[-1] = output_bias
            simple_model.set_weights(weights)
        
        # Set this as our active model
        self.model = simple_model
        self.target_model = simple_model
        self.trained = True
        
        # Log the baseline model creation
        logging.info("Baseline model created successfully")
        
        # Save the baseline model
        os.makedirs('models', exist_ok=True)
        baseline_path = 'models/baseline_model.h5'
        simple_model.save(baseline_path)
        logging.info(f"Baseline model saved to {baseline_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to create baseline model: {str(e)}")
        return False

# Monkey patch the ReinforcementTrainer class
from ai_optimization.reinforcement_trainer import ReinforcementTrainer
ReinforcementTrainer.create_baseline_model = create_baseline_model
