# Experiment config file

# Experiment configuration
experiment_name = "dist3"
experiments_path= "/home/willytell/Experiments"
experiment_prefix = "experiment"
num_repetition_experiment = 10      # Repeat n times the experiment

# Cross validation
num_images_for_test = 24
n_splits = 5
n_splits_prefix = "kfold"

num_classes = 2

# Dataset conf
dataset_division_strategy = "keep-unbalanced" # keep unbalanced rates of each classes from the original dataset
dataset_prefix = "dataset"
dataset_directory = "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN"
images_filenames = "images_filenames.txt"
labels = "binary_classification.txt"
#dataset_images_path = "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/Original"   # all images are in this folder
dataset_images_path = "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/BBox"   # all images are in this folder

# Generate images using bounding boxes of masks.
dataset_mask_directory = "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/GT"
bbox_output_path       = "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/BBox"    # directory to write the images from mask


# Model
model_name                   = 'resnet50'
freeze_layers_from           = None            # Freeze layers from 3 to this layer during training (Useful for finetunning) [None | 'base_model' | Layer_id]
show_model                   = False
load_imageNet                = False           # Load Imagenet weights and normalize following imagenet procedure
load_pretrained              = False
weights_suffix               = 'weights.hdf5'  # Pre-Trained weight file name

# Batch sizes
batch_size_train = 8            # Batch size during training
batch_size_valid = 3            # Batch size during validation
batch_size_test  = 6            # Batch size during testing

resize_image                  = (224, 224)       # Resize the image during training (Height, Width) or None
#resize_train                 = (224, 224)      # Resize the image during training (Height, Width) or None
#resize_valid                 = (224, 224)      # Resize the image during validation
#resize_test                  = (224, 224)      # Resize the image during testing

crop_size_image               = None       # [(224, 224) | None]
#crop_size_train              = None       # Crop size during training (Height, Width) or None
#crop_size_valid              = None       # Crop size during validation
#crop_size_test               = None       # Crop size during testing


# Data shuffle
shuffle_train                = False     # Whether to shuffle the training data
shuffle_valid                = False     # Whether to shuffle the validation data
shuffle_test                 = False     # Whether to shuffle the testing data
seed_train                   = 1925      # Random seed for the training shuffle
seed_valid                   = 1925      # Random seed for the validation shuffle
seed_test                    = 1925      # Random seed for the testing shuffle

batch_shuffle_train          = True      # Whether to shuffle the (mini) batch training data 
batch_shuffle_valid          = False     # Whether to shuffle the (mini) batch valid data 
batch_shuffle_test           = False     # Whether to shuffle the (mini) batch test data 
seed_batch_train             = 55
seed_batch_valid             = 55
seed_batch_test              = 55

# Training parameters
optimizer = 'adam'
learning_rate = 0.001
n_epochs = 200

# Normalization and Standardize
norm_rescale                       = 1/255.    # Scalar to divide and set range 0-1
norm_featurewise_center            = True      # Substract mean - dataset
norm_featurewise_std_normalization = True      # Divide std - dataset


# Callback TensorBoard
memory_usage_enabled         = False    # Print main memory RAM usage at the end of each epoch.
TensorBoard_enabled          = True     # Enable the Callback
TensorBoard_histogram_freq   = 0        # Frequency (in epochs) at which to compute activation histograms for the layers of the model. If set to 0, histograms won't be computed.
TensorBoard_write_graph      = True     # Whether to visualize the graph in Tensorboard. The log file can become quite large when write_graph is set to True.
TensorBoard_write_images     = False    # Whether to write model weights to visualize as image in Tensorboard.

# Callback early stoping
earlyStopping_enabled        = True            # Enable the Callback
earlyStopping_monitor        = 'val_loss'      # Metric to monitor
earlyStopping_mode           = 'min'           # Mode ['max' | 'min']
earlyStopping_patience       = 40              # Max patience for the early stopping
earlyStopping_verbose        = 1               # Verbosity of the early stopping

# Callback reduce LR on plateau
reduceLROnPlateau_enabled    = True            # Enable the Callback
reduceLROnPlateau_monitor    = 'val_loss'      # Metric to monitor
reduceLROnPlateau_factor     = 0.8             # Factor by which the learning rate will be reduced. new_lr = lr * factor
reduceLROnPlateau_mode       = 'auto'          # Mode ['auto' | 'min' | 'max']
reduceLROnPlateau_patience   = 10              # Number of epochs with no improvement after which learning rate will be reduced
reduceLROnPlateau_cooldown   = 5               # Number of epochs to wait before resuming normal operation after lr has been reduced
reduceLROnPlateau_min_lr     = 0.00001         # Lower bound on the learning rate
reduceLROnPlateau_verbose    = 1               # int. 0: quiet, 1: update messages
reduceLROnPlateau_epsilon    = 0.0001

# Callback model check point
checkpoint_enabled           = True            # Enable the Callback
checkpoint_monitor           = 'val_acc'       # Metric to monitor
checkpoint_mode              = 'max'           # Mode ['max' | 'min']
checkpoint_save_best_only    = False           # Save best or last model
checkpoint_save_weights_only = True            # Save only weights or also model
checkpoint_verbose           = 1               # Verbosity of the checkpoint
checkpoint_filename          = 'weights-{epoch:002d}-{val_acc:.4f}.hdf5'  # weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5 | weights.hdf5

# Data augmentation for training
apply_augmentation           = True 
n_augmentation               = 10

da_rotation_range            = 30         # Rnd rotation degrees 0-180
da_width_shift_range         = 0.01       # Rnd horizontal shift
da_height_shift_range        = 0.01       # Rnd vertical shift
da_shear_range               = 0.02       # Shear in radians
da_zoom_range                = [1, 1]     # Zoom
da_channel_shift_range       = 0.         # Channecf.l shifts
da_fill_mode                 = 'constant' # Fill mode ['constant' | 'nearest']
da_cval                      = 0.         # Void image value
da_horizontal_flip           = True       # Rnd horizontal flip
da_vertical_flip             = True       # Rnd vertical flip
da_spline_warp               = True       # Enable elastic deformation
da_warp_sigma                = 0.01       # Elastic deformation sigma
da_warp_grid_size            = 1          # Elastic deformation gridSize
da_save_to_dir               = False      # Save the images for debuging



model_output_directory = 'resnet50-from-scratch-bbox-without-imagenet-with-da-exp9'
