# Notebook Setup
- **Important**: Notebook generates large files into /Garbage_data folder.
- ⚠️ Current notebook size ~ **4.5 GB**

- GarbageClean.ipynb is depreciated and should not be used for model execution or testing. Updated Preprocessing is now in PreProcessAndFeatureExtraction.ipynb.

- **Execution Flow**:
    1. Load and execute PreProcessAndFeatureExtraction.ipynb
        - Downloads dataset from Kaggle
        - Uses metadata to split dataset and form file paths for each sample
        - Performes **Feature Extraction** of HOG and LBP of resized, blurred, and grayscale samples
        - Saves all modified datasets as feather files in /Garbage_data folder
    2. Load and execute PipelineAndTrain.ipynb
        - Loads modified datasets
        - Allows for full dataset CV evaluation or train/validation split training and testing
        - Initializes Sklearn pipelines (SVM, RF)
        - Trains models and plots accuracy over different training sizes up to 80/20 split. (sample size 1.00 correspondes to 80% train and 20% validation)
        - Calulates 95% Confidence Intervals for each training set size
    3. dataAnalysis.ipynb is stand alone and can be executed at any time after Kaggle dataset is downloaded.



## Required Modules
This project requires the follow Python Packages.

    - kaggle
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - opencv-python
    - seaborn
    - scikit-image

## For Questions
contact: mfecco@vols.utk.edu
