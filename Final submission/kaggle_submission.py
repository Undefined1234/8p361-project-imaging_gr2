'''
TU/e BME Project Imaging 2021
Submission code for Kaggle PCAM
Author: Suzanne Wetstein
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   

import numpy as np
import glob
import pandas as pd
from matplotlib.pyplot import imread
from tensorflow.keras import backend
from main_util import Model_architecture

#Change these variables to point at the locations and names of the test dataset and your models.
model_name = "cnn_augmented_1"

TEST_PATH = 'C:/Users/20212077/OneDrive - TU Eindhoven/Desktop/8P361 - DBL AI for MIA/8p361-project-imaging_gr2/data/test' 
WEIGHTS_FILEPATH = f"C:/Users/20212077/OneDrive - TU Eindhoven/Desktop/8P361 - DBL AI for MIA/8p361-project-imaging_gr2/Final submission/trained_models/{model_name}_weights.hdf5"

# load weights into new model
model = Model_architecture()
model.create_cnn(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64)
model.compile_cnn(learning_rate=0.001)

model.load_weights(WEIGHTS_FILEPATH)

# open the test set in batches (as it is a very big dataset) and make predictions
test_files = glob.glob(os.path.join(TEST_PATH, '*.tif'))

submission = pd.DataFrame()

file_batch = 1000
max_idx = len(test_files)

for idx in range(0, max_idx, file_batch):

    print('Indexes: %i - %i'%(idx, idx+file_batch))

    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})

    # get the image id 
    test_df['id'] = test_df.path.map(lambda x: x.split(os.sep)[-1].split('.')[0])
    test_df['image'] = test_df['path'].map(imread)
    
    K_test = np.stack(test_df['image'].values)
    
    # apply the same preprocessing as during draining
    K_test = K_test.astype('float')/255.0
    
    predictions = model.predict(K_test)
    
    test_df['label'] = predictions
    submission = pd.concat([submission, test_df[['id', 'label']]])
    backend.clear_session()

# save your submission
submission.head()
submission.to_csv(f'submission_{model_name}.csv', index = False, header = True)
