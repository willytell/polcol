import os
from importlib.machinery import SourceFileLoader

class Configuration():
    def __init__(self, config_path, action):
        self.config_path = config_path
        self.action = action

    def load(self):
        # load experiment config file
        cf = SourceFileLoader('config', self.config_path).load_module()

        #  Divide the dataset
        if self.action == 'divide':

            # Follow a division strategy
            if cf.dataset_division_strategy == 'keep-unbalanced':
                # create experiment paths
                cf.output_path = os.path.join(cf.experiments_path, cf.experiment_name)

                if not os.path.exists(cf.output_path):
                    os.makedirs(cf.output_path)


        # Train the cnn
        if self.action == 'train':
            print("Config for train...")

            cf.output_path = os.path.join(cf.experiments_path, cf.experiment_name, cf.model_output_directory)

            if not os.path.exists(cf.output_path):
                os.makedirs(cf.output_path)


        # Test and prediction, include confusion matrix
        if self.action == 'test':
            print("Config fo test...")

            # create the 'predictions' directory


        return cf