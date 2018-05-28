import math
import os
import resource

from keras.callbacks import TensorBoard, Callback, ModelCheckpoint

class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        #print("\n > Memory: maximum resident set size used {} MB.", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
        print("\n > Main memory RAM size used: {:.2f} GB.".format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024))

# Create callbacks
class Callbacks_Factory():
    def __init__(self):
        pass

    def make(self, cf, tensorboard_path, modelcheckpoint_path, modelcheckpoint_fname):
        cb = []

        # Memory usage
        if cf.memory_usage_enabled:
            print('   Memory usage')
            cb += [MemoryCallback()]

        # TensorBoard callback
        if cf.TensorBoard_enabled:
            print('   Tensorboard')

            if not os.path.exists(tensorboard_path):
                os.makedirs(tensorboard_path)
            cb += [TensorBoard(log_dir=tensorboard_path,
                               histogram_freq=cf.TensorBoard_histogram_freq,
                               write_graph=cf.TensorBoard_write_graph,
                               write_images=cf.TensorBoard_write_images)]

        # Define model saving callbacks
        if cf.checkpoint_enabled:
            print('   Model Checkpoint')
            cb += [ModelCheckpoint(filepath=os.path.join(modelcheckpoint_path, modelcheckpoint_fname),
                                   verbose=cf.checkpoint_verbose,
                                   monitor=cf.checkpoint_monitor,
                                   mode=cf.checkpoint_mode,
                                   save_best_only=cf.checkpoint_save_best_only,
                                   save_weights_only=cf.checkpoint_save_weights_only)] 


        # Output the list of callbacks
        return cb
