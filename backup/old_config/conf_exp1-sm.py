# Experiment config file

# Experiment configuration
experiment_name = "exp1-whole_image"
experiments_path= "/imatge/mgorriz/work/Guillermo/Experiments"
experiment_prefix = "experiment"
num_repetition_experiment = 1      # Repeat n times the experiment

# Cross validation
num_images_for_test = 24
n_splits = 5 
n_splits_prefix = "kfold"

num_classes = 2

# Dataset conf
dataset_division_strategy = "keep-unbalanced" # keep unbalanced rates of each classes from the original dataset
dataset_prefix = "dataset"
dataset_directory = "/imatge/mgorriz/work/Guillermo/ExperCNN"
images_filenames = "images_filenames.txt"
labels = "binary_classification.txt"
dataset_images_path = "/imatge/mgorriz/work/Guillermo/ExperCNN/Original"   # all images are in this folder

# Model
model_name = 'vgg16'
load_pretrained = False
weights_suffix = 'weights.hdf5'  # Pre-Trained weight file name
show_model = False

# Batch sizes
batch_size_train = 8            # Batch size during training
batch_size_valid = 8            # Batch size during validation
batch_size_test  = 6            # Batch size during testing

resize_image = (224, 224)       # Resize the image during training (Height, Width) or None
#resize_train = (224, 224)      # Resize the image during training (Height, Width) or None
#resize_valid = (224, 224)      # Resize the image during validation
#resize_test  = (224, 224)      # Resize the image during testing

# Data shuffle
shuffle_train = True            # Whether to shuffle the training data
shuffle_valid = True            # Whether to shuffle the validation data
shuffle_test  = False           # Whether to shuffle the testing data


# Training parameters
optimizer = 'rmsprop'
learning_rate = 0.0001
n_epochs = 2

# Normalization and Standardize
norm_rescale                       = 1/255.    # Scalar to divide and set range 0-1
norm_featurewise_center            = True      # Substract mean - dataset
norm_featurewise_std_normalization = True      # Divide std - dataset

# Callback TensorBoard
memory_usage_enabled         = False    # Print memory usage at the end of each epoch.
TensorBoard_enabled          = True     # Enable the Callback
TensorBoard_histogram_freq   = 0        # Frequency (in epochs) at which to compute activation histograms for the layers of the model. If set to 0, histograms won't be computed.
TensorBoard_write_graph      = True     # Whether to visualize the graph in Tensorboard. The log file can become quite large when write_graph is set to True.
TensorBoard_write_images     = False    # Whether to write model weights to visualize as image in Tensorboard.


# Callback model check point
checkpoint_enabled           = True            # Enable the Callback
checkpoint_monitor           = 'acc'           # Metric to monitor
checkpoint_mode              = 'max'           # Mode ['max' | 'min']
checkpoint_save_best_only    = True            # Save best or last model
checkpoint_save_weights_only = True            # Save only weights or also model
checkpoint_verbose           = 0               # Verbosity of the checkpoint



model_output_directory = 'vgg-from-scratch'

