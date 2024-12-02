# Hydrological-concepts-inside-LSTM
This model is adapted from the traditional LSTM model so that during training, the model's state variables are mapped to hydrological concepts (such as soil moisture and snow depth) in the runoff formation process through a linear layer.
The source code provided is written in Python programming language and has been tested using Python 3.7. The main libraries used in this project include tensorflow2.6.0 and Keras2.6.0. For specific code environment settings, please refer to environment.yml, which can be used to create a new virtual environment.
- - - -
SYSTEM REQUIREMENTS:
Please make sure that the following Python packages are installed on your computer before running any of the above execution files:
(1) NumPy (http://www.numpy.org/)
(2) SciPy (http://www.scipy.org/)
(3) matplotlib (http://matplotlib.org/)
- - - -
DATA RESOURCES:
CAMELS: Catchment Attributes and Meteorology for Large-sample Studies (https://ral.ucar.edu/solutions/products/camels)
Caravan - A global community dataset for large-sample hydrology (https://www.nature.com/articles/s41597-023-01975-w)
ERA5_Land data was obtained through Google Earth Engine (https://www.ecmwf.int/en/era5-land)
's_391attrs.csv' contains the normalized static attribute data of the watershed
- - - -
CODE:

Please set your own data and model storage path before running the following executable files:

(1)'LSTM_train.py': This file is used for model training. Some parameters can be adjusted according to actual needs.

(2)'LSTM_test.py': This file is used to test the model. The test time period and performance indicators can be selected according to research needs.

(3)'LSTM_output_Cn.py': This file is used to output the cell state of the model. The dimension of the cell state should be consistent with the dimension set when the model is created.

(4)'hydrodata_tf.py': This file is mainly used for data preprocessing of the CAMELS database, including unit conversion and feature concatenation.

(5)'libs_tf.py': This file contains a custom LSTM model and a linear regression layer for training probes.

(6)'cn_loss.py': This file contains the Loss function and monitoring indicators used when training the model in this project.
