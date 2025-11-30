import tensorflow as tf

# List all physical GPUs visible to TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available:", gpus)