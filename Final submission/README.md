# Code Project AI for MIA - group 2

## Installation

The Python version used is 3.9.18. The following libraries were used including their versions:

    - matplotlib 3.8.0
    - numpy 1.26.4
    - jupyter 1.0.0
    - scikit-learn 1.3.0
    - scipy 1.11.4
    - tensorflow 2.10.1 (not most recent due to GPU-support!)  
    - cudatoolkit 11.2
    - cudnn 8.1.0

An environment can be made in the following way in order to correctly load the models into the notebooks:

    conda create --name group2 python=3.9.18
    conda activate group2
    conda install matplotlib jupyter scikit-learn scipy spyder pandas
    pip install "tensorflow<2.11"
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

NOTE: if the environment is not set up correctly, there could be problems with loading the models.


## Usage

The code consists of three Jupyter Notebooks and two Python files. Their usage is explained below:

1. main_util.py -> Python file containing general classes and functions that are used in the project notebooks.

    - get_pcam_generators: function used for setting up data generators to load the data in batches into the memory.
    - Model_architecture: class used for building the model architectures of the convolutional neural networks and the autoencoder
    - Model_transform: class used as preprocessing function in the data generators for the augmentation of training data 
    - evaluation_pcam17: function used for obtaining the true labels and predicted labels on the testing set of the pcam17 dataset.

2. main_manual_training.ipynb -> Notebook with code to train all the deep learning models.

3. main_project_notebook.ipynb -> Notebook with code for ROC-curve analysis and evaluation of the trained models on the validation set.

4. domain_generalization.ipynb -> Notebook with code for ROC-curve analysis of the trained models on the pcam17 dataset.

5. kaggle_submission.py -> Python file containing code to generate the submission .csv files, which were submitted to Kaggle to obtain testing set AUC values.


## Contributors

Constantijn bok
Nino Jacobs
Kasper Wessels
Mart van Straten
