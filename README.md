# Credit_Risk_Classifier
A machine learning program, used to identify the creditworthiness of borrowers using  machine learning algorthims. The Machine Learning algorithm is a supervised learning algorithm, which will be used in-tangent with resampling methods to oversample the underlying training data. 

##  Background 

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Challenge, you’ll use various techniques to train and evaluate models with imbalanced classes. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

## Instructions

  - This challenge consists of the following subsections:

  - Split the Data into Training and Testing Sets

  - Create a Logistic Regression Model with the Original Data

  - Predict a Logistic Regression Model with Resampled Training Data

  - Write a Credit Risk Analysis Report


## Technologies

### Instructions

 This project uses the software jupyter lab with the following  softwares, and their installed versions :
 
  - Jupyter Lab - 3.4.4
  - Python - 3.10.4

The following python modules are also used in the application. Remember to install these packages via Terminal for MacOS/Linux or GitBash for windows clients. 

  - * [numpy](https://github.com/numpy/numpy)
    - NumPy is the fundamental package for scientific computing with Python. It provides: a powerful N-dimensional array object, sophisticated (broadcasting) functions tools for integrating C/C++ and Fortran code, useful linear algebra, Fourier transform, and random number capabilities. 
  - * [pandas](https://github.com/pandas-dev/pandas) - pandas is used to interact with data packages, plot data frames, create new dataframes, describe abailable data, and helps traders and fintech proffesionals organize financial data to perform advanced decisionmaking. 
  - * [pathlib](https://github.com/python/cpython/blob/main/Lib/pathlib.py) - Allows the user to specify the path to a data frame / any data in a csv file. 
  - * [sklearn.metrics]([https://github.com/python/cpython/blob/main/Lib/pathlib.p](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_classification.py)y) - scikit-learn is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license.
  - * [warnings]([https://docs.python.org/3/library/warnings.html]) - Warning messages are typically issued in situations where it is useful to alert the user of some condition in a program, where that condition (normally) doesn’t warrant raising an exception and terminating the program. For example, one might want to issue a warning when a program uses an obsolete module.
  - * [imblearn]([https://pypi.org/project/imbalanced-learn/]) - imbalanced-learn is a python package offering a number of re-sampling techniques commonly used in datasets showing strong between-class imbalance. It is compatible with scikit-learn and is part of scikit-learn-contrib projects.
  - * [pyplotplus]([https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/pyplot.py]) - `matplotlib.pyplot` is a state-based interface to matplotlib. It provides
an implicit,  MATLAB-like, way of plotting.  It also opens figures on your
screen, and acts as the figure GUI manager.


## Installation Guide


There are four installations neccesary for the program to run. Install all modules and libraries neccesary by:

1. first activating your conda dev environment via terminal (for MacOS) or GitBash (Windows/Linux) command :

 - ''' conda activate dev'''


2. After activating the dev environment, install imbalance-learn by running the following commands:


 - 'conda install -c conda-forge imbalanced-learn'
 -  &
 -  'conda install -c conda-forge pydotplus'

3. Verify the installation by running the following commands:

  - 'conda list imbalanced-learn'
  - & 
  - 'conda list pydotplus'

4. Finally, run the following commands to download the rest of the python modules (common modules) :

    '''
    pip3 install pandas
    pip3 install numpy
    pip3 install pathlib
    '''


After activating the dev environment, install the following libraries via. the command line :

'''python
    pip3 install pandas
    pip3 install numpy
    pip3 install pathlib
    pip install pyplotpluw
'''

## Usage

This application can be accessed by viewing the {{credit_risk_resampling.ipynb}} file using jupyter notebook. It contains information on how to setup a similar application, and so the application can be used as a guide for anyone looking to understand how categorizer models work, can be optimized, and can be evaluated for their accuracy. 

## Contributors

The sole contributor for this project is:

**NAJIB ABOU NASR**
 no instagram or linkedin yet!
---

## License

Using the 'MIT' license!
--- 

## History

### 
    Here, I documented my command line inputs, to show the changes I had made, and document the debugging and programing process:  
---
