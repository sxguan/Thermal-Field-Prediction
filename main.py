# Author: Shixuan Guan(@sxguan)
# Date: 12/14/2024

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# Data file name
data_file = 'laplace_results_v3.csv'

# Define plane size
grid_size = 100

data_min = 0.0
data_max = 100.0

train_mode = False
# Create boundary and internal masks
def create_boundary_mask():
    mask = np.zeros((grid_size, grid_size, 1), dtype=np.float32)
    mask[0, :, 0] = 1   # top boundary
    mask[-1, :, 0] = 1  # bottom boundary
    mask[:, 0, 0] = 1   # left boundary
    mask[:, -1, 0] = 1  # right boundary
    return tf.constant(mask)

def create_internal_mask():
    return 1 - create_boundary_mask()

boundary_mask = create_boundary_mask()
internal_mask = create_internal_mask()

# Parse CSV file to extract features (boundary temperature) and labels (internal temperature)
def parse_csv_line(line):
    record_defaults = [[0.0]] * (grid_size * grid_size)
    fields = tf.io.decode_csv(line, record_defaults=record_defaults)
    full_grid = tf.reshape(tf.stack(fields), (grid_size, grid_size, 1))
    full_grid = tf.cast(full_grid, tf.float32)

    # Normalize data to [0, 1]
    full_grid = (full_grid - data_min) / (data_max - data_min)

    # Features: boundary part
    features = tf.multiply(full_grid, boundary_mask)
    # Labels: internal part
    labels = tf.multiply(full_grid, internal_mask)

    return features, labels

# Create dataset and skip the header line
dataset = tf.data.TextLineDataset(data_file).map(parse_csv_line)

# Dataset splitting
dataset = dataset.shuffle(buffer_size=1000)
total_samples = sum(1 for _ in open(data_file))  # total lines in file
train_size = int(0.7 * total_samples)
val_size = int(0.2 * total_samples)

batch_size = 32
train_dataset = dataset.take(train_size).batch(batch_size)
val_dataset = dataset.skip(train_size).take(val_size).batch(batch_size)
test_dataset = dataset.skip(train_size + val_size).batch(batch_size)

# Define U-Net model
def unet_model(input_shape=(grid_size, grid_size, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder part
    # down 1
    c1 = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # down 2
    c2 = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')(c3)

    # Decoder part
    # up 1
    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(c4)

    # up 2
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(c5)

    # Output layer
    outputs = layers.Conv2D(1, kernel_size=3, activation='linear', padding='same')(c5)

    # Restore actual scale: predict internal + preserve boundary
    # Since we normalized data to [0,1], predicted outputs should also be in [0,1]
    # Apply mask to output to ensure boundary remains unchanged and internal is predicted
    # Boundary comes from inputs
    outputs = outputs * internal_mask + inputs * boundary_mask

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()

# Custom loss function: only compute error for internal temperature
def masked_mse(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred) * internal_mask
    return tf.reduce_mean(squared_difference)

# Custom metric
def root_mean_squared_error(y_true, y_pred):
    mse = masked_mse(y_true, y_pred)
    return tf.sqrt(mse)

# Tolerance accuracy metric
class ToleranceAccuracy(tf.keras.metrics.Metric):
    def __init__(self, tolerance=0.01, name='tolerance_accuracy', **kwargs):
        super(ToleranceAccuracy, self).__init__(name=name, **kwargs)
        self.tolerance = tolerance
        self.total = self.add_weight(name='total', initializer='zeros')
        self.correct = self.add_weight(name='correct', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_internal = y_true * internal_mask
        y_pred_internal = y_pred * internal_mask

        difference = tf.abs(y_true_internal - y_pred_internal)
        correct_predictions = tf.cast(tf.less_equal(difference, self.tolerance), tf.float32)

        total_predictions = tf.reduce_sum(tf.ones_like(correct_predictions))
        self.correct.assign_add(tf.reduce_sum(correct_predictions))
        self.total.assign_add(total_predictions)

    def result(self):
        return self.correct / self.total

    def reset_states(self):
        self.total.assign(0)
        self.correct.assign(0)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,  # initial learning rate
    decay_steps=1000,            # steps after which learning rate decays
    decay_rate=0.9               # decay rate
)

# Define optimizer with a learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# Compile the model
model.compile(
    optimizer='adam',
    #optimizer=optimizer,
    loss=masked_mse,
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(),
        root_mean_squared_error,
        ToleranceAccuracy(tolerance=0.1)  # Adjust tolerance for normalized data
    ]
)

model.summary()

if train_mode:
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model_v3.h5', save_best_only=True)
    ]

    # Start training
    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=callbacks
    )
    end_time = time.time()

    # Print training time
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    # Save the model
    model.save('best_model_v3.h5')

    # Plot training and validation loss and MAE
    # Loss curve
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # MAE curve
    plt.figure()
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('MAE Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

else:
    # Load trained model
    model = tf.keras.models.load_model(
        'best_model_v3.h5',
        custom_objects={
            'masked_mse': masked_mse,
            'root_mean_squared_error': root_mean_squared_error,
            'ToleranceAccuracy': ToleranceAccuracy(tolerance=0.1)
        }
    )

# Evaluate on test set
test_results = model.evaluate(test_dataset, return_dict=True)
print(f"Test Results:")
for metric_name, value in test_results.items():
    print(f"- {metric_name}: {value:.4f}")

# Visualization of prediction results vs. true values
def plot_temperature_distribution(true_grid, predicted_grid):
    # De-normalize: map [0,1] back to original scale
    true_grid = true_grid.numpy().reshape(grid_size, grid_size)
    predicted_grid = predicted_grid.reshape(grid_size, grid_size)

    true_grid = true_grid * (data_max - data_min) + data_min
    predicted_grid = predicted_grid * (data_max - data_min) + data_min

    plt.figure(figsize=(12, 6))

    # Fix the colorbar range from 0 to 100
    vmin = 0
    vmax = 100

    # True temperature distribution
    plt.subplot(1, 2, 1)
    plt.imshow(true_grid, cmap='hot', vmin=vmin, vmax=vmax)
    plt.title('True Temperature Distribution')
    plt.colorbar()

    # Predicted temperature distribution
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_grid, cmap='hot', vmin=vmin, vmax=vmax)
    plt.title('Predicted Temperature Distribution')
    plt.colorbar()

    plt.show()

# Select one sample from the test set for visualization
for features, labels in test_dataset.take(1):
    predictions = model.predict(features)
    for i in range(1):  # visualize the first sample
        true_grid = features[i] + labels[i]
        predicted_grid = predictions[i]
        smooth_predicted_grid = gaussian_filter(predicted_grid, sigma=2.0)
        plot_temperature_distribution(true_grid, smooth_predicted_grid)
    break
