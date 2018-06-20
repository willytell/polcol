# Experiment config file

# Experiment configuration
experiment_name = "exp1-whole_image"
experiments_path= "C:\\Users\\guille\\Experiments"
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
dataset_directory = "C:\\Users\\guille\\Documents\\MCV\\M9\\TFM\\ExperCNN"
images_filenames = "images_filenames.txt"
labels = "binary_classification.txt"
#dataset_images_path = "/home/willytell/Documentos/MCV/M9/TFM/ExperCNN/Original"   # all images are in this folder
dataset_images_path = "C:\\Users\\guille\\Documents\\MCV\\M9\\TFM\\ExperCNN\\BBox"   # all images are in this folder

# Generate images using bounding boxes of masks.
dataset_mask_directory = "C:\\Users\\guille\\Documents\\MCV\\M9\\TFM\\ExperCNN\\GT"
bbox_output_path       = "C:\\Users\\guille\\Documents\\MCV\\M9\\TFM\\ExperCNN\\BBox"    # directory to write the images from mask


# Model
model_name                   = 'resnet50'
show_model                   = False
load_imageNet                = True            # Load Imagenet weights and normalize following imagenet procedure
load_pretrained              = False
weights_suffix               = 'weights.hdf5'  # Pre-Trained weight file name

# Batch sizes
batch_size_train = 8            # Batch size during training
batch_size_valid = 3            # Batch size during validation
batch_size_test  = 6            # Batch size during testing

resize_image                  = (224, 224)      # Resize the image during training (Height, Width) or None
#resize_train                 = (224, 224)      # Resize the image during training (Height, Width) or None
#resize_valid                 = (224, 224)      # Resize the image during validation
#resize_test                  = (224, 224)      # Resize the image during testing

crop_size_image               = None       # [(224, 224) | None]
#crop_size_train              = None       # Crop size during training (Height, Width) or None
#crop_size_valid              = None       # Crop size during validation
#crop_size_test               = None       # Crop size during testing


# Data shuffle
shuffle_train       = False     # Whether to shuffle the training data
shuffle_valid       = False     # Whether to shuffle the validation data
shuffle_test        = False     # Whether to shuffle the testing data
batch_shuffle_train = True      # Whether to shuffle the (mini) batch training data 
batch_shuffle_valid = False     # Whether to shuffle the (mini) batch valid data 
batch_shuffle_test  = False     # Whether to shuffle the (mini) batch test data 

# Training parameters
optimizer = 'adam'
learning_rate = 0.1
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


# Callback model check point
checkpoint_enabled           = True            # Enable the Callback
checkpoint_monitor           = 'val_acc'       # Metric to monitor
checkpoint_mode              = 'max'           # Mode ['max' | 'min']
checkpoint_save_best_only    = True            # Save best or last model
checkpoint_save_weights_only = True            # Save only weights or also model
checkpoint_verbose           = 1               # Verbosity of the checkpoint

# Data augmentation for training
apply_augmentation           = True 
n_augmentation               = 20

da_rotation_range            = 5          # Rnd rotation degrees 0-180
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



model_output_directory = 'resnet50-from-scratch-bbox-da-class_weights-imagenet-francho'
