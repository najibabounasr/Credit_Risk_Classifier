# Credit_Risk_Classifier
A machine learning program, used to identify the creditworthiness of borrowers using  machine learning algorthims. The Machine Learning algorithm is a supervised learning algorithm, which will be used in-tangent with resampling methods to oversample the underlying training data. 

##  Background 

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Challenge, you’ll use various techniques to train and evaluate models with imbalanced classes. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

# Module 12 Report 

## Overview / Analysis

In this section, I describe the analysis completed using the machine learning protocols in this challenge, and how it applies to the real-world situations. 


To start off, the purpose of the program was to identify whether a
proposed loan would result in a positive, or negative outcome. In the case of our dataset, the negative outcome would be a loan default, whereas a positive outcome would result from a succesful loan, payed in full. The program use machine learning algorithms, specifically a supervised learning algorithm, trained with a labelled dataset of financial information, to prepare it to correctly identify profitable loans from those that result in default. In our dataset, 'profitable loans' are labelled as '0', while 'defaults' are labelled as '1'. 
The data included informartion on the loan size, the interest rate, the borrowers income, the debt-to-income, the number of accounts, the 'derogatory marks', and the 'total debt' of potential debtors. As creditors, we look to isolate all debtors who will pay their dues, and all those who will not. 
The machine learning program used, is a supervised learning program, meaning that it is capable of classifying records in a dataset into different categories, based on past 'labelled' datasets we have given it as training material. The terminology for this training is known as 'fitting'-- you fit a 'malleable model' to a labelled dataset, which is a datset containing entries of data you want to predict, with related variables aswell. For example, if you wanted to predict 'BMI' based on calories, calorie expenditure, and age, you would feed a model all these variables, but specify to it which specific 'target variable' it will be predicting for, and which variables are to be used to help in identifying the target variable. 
As this is a general summary, I will not go into detail about the specfics of the machine learning protocol-- but will explain the steps that are neccesary to reach a conclusion : 

  - 1. First, we find the data. The data should include any variables we know (or can reasonably assume) will help the model predict for the 'target variable' -- the variable we looking to predict. Thus, this step takes place before creating the model, and may include normalization or reduction of 'noise' in other cases. In the case of this challenge, I inherited the data from Berkley. 
  - 2. Second, we look to split the data into training and testing sets , so that we can train the model using the training set (known as 'fitting'), and later test the results. 
  - 3. We may check for the balance of the target variable 'labels variable' using the sklearn function {{value_counts}}. This function will allow us to see if there is an imbalance within the dataset. For example, if we are looking to train a model to differentiate PC gamers from console gamers based on game-data, we won't be as succesfull if we have a 9:1 ratio of PC gamers to console gamers. In the case of our data we identified a clear majority being the 'succesful loans' or '0' values. This is to be expected, and is completely normal. 
  - 4. In order to correct the bias in our data, we can artifically do so using 'resampling' techniques. I won't get into detail here, but resampling is done by either 'oversampling' the minority category, or 'undersampling' the majorty category to correct the massive difference in values. The resampling is important so that the model does not recieve a great accuracy rate, or a bias to identifying one category rather than the other later during testing. Therefore, we use the {{RandomOverSampler}} function to correct the bias, and therefore optimally train our model. 
  - 5. Though this could be done before resampling, and resampling is not always neccesary, the next step would be Logistic Regression. We use the function LogisticRegression to create our malleable, or 'empty' model. 
  - 6. After creating the model, we train the model. by fitting it to the data. in our case, to the resampled data aswell as the original. When resampling, we still mjstevaluate the original data, to understand if resampling has improved our results, as this definetely may not be the case. 
  - 7. Finally, after fitting, we must evaluate the performance of the model. In our program, we utilize the {{balanced_accuracy_score}} to identify the accuracy score of our model. {{balanced_accuracy_score}} is different from {{accuracy_score}} in that is takes into account imbalanced in our target variable (the discrepency between defaults, and succesful loans). Another method of evaluation is using the {{confusion_matrix}} function, which creates a useful table identifying the fale positive, false negative, true positive, and true negative predictions of our model. Finally, we use the function {{classification_report_imbalanced}} to identify numerous metrics that relate to our data. Of these, precision, recall, accuracy, and the f-1 score stand out as key identifiers of the models performance. The many metrics will be discussed in further detail in the results section below. The three functions I have mentioned, all use specific mathematical techniques, which are commonly used by data scientists to 'make sense' of data. Understanding the model and it;s underlying data, which drives it, is the most important feature of the developer who wishes to utilize machine learning. 
    

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

  - To my surprise, the model does seem to be able to predict both the healthy loans, and the high-risk loans quite well. Specifically, the model seems to be more conservative in 'incorrectly' identifying a healthy loan-- as it has a precision of 1 in terms of identifying the healthy loans. Depending on the business, and how it wishes to run (and what protection it has from suffering from loan defaults), this may be the optimal situation, or be suboptimal.

  - Furthermore, we can see that the recall for both classifications of loans is higher for 'type 0', which is to be expected, as type 0 is the 'majority' classifier in this instance. We can infer from the lower precision, and higher recal of type '0' that the model seems to have adopted a strategy of prioritizing 'never labeling a good loan as a bad loan', rather than focus too much on 'faulty loans'. 

  - Here, we have a great example of a model where 'accuracy doesn't tell us everything'. This is best demonstrated by the two f-1 scores : '0' loans have an f-1 score of 1, while the '1' loans get an 85. The model had achieved similar levels of accuracy in identifying both, but what this may tell us is that the model has enough data on '0' loans to almost perfectly identify when a loan will be classified as '0', but because of it's lack of knowledge on '1' loans, it has began making up by incorrectly identifying numerous '0' loans to boost it's stats. This can be inferred by the models extremely high f-1 score, and its high level of precision-- meaning, unless 100% sure a loan will be '0', it will just go ahead and label it as a '1'. This is not an optimal solution, as the data shows that it is still suffering in terms of overall accuracy, due to the imbalance in criteria. 


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

  The accuracy of the model has improved immensly!, I will break down the improvements in steps :

    - First of all, the model has improved in overall accuracy, from an average accuracy of 0.91, to an overall accuracy of around 0.99. This improvement for both of the loans means that, as a whole, the model is now correctly identifying a 'greater amount' of both loan subtypes.

    Yet, when looking to the precision, we see a strange trend -- we have identified that the model will now have a higher tendency to 'miscategorize' 'good loans' as 'loans that will default'. It has actually recieved a worse precision score for loan types (1) -- a change of -0.01. This does not tell the whole story, of course.

  - Finally, the model seems to have improved immensly in it's 'recall' of loans which result in default. A high recall rate, and a lower precision singify that the model has adopted an 'aggressive' strategy -- risking 'mislabeling' some '0' loans as '1's, in favor of catching around '99%' of the '1' loans ('1's result in default). From a business standpoint, this strategy may very well make alot of sense -- as many of the '0's miscategorized would have likely ben 'on the fringe'-- as the model has an almost perfect record of precisely identifying '0' loans.

  We may look to the confusion matrix for further information. 4 'bad' (riskier) loans had fallen under the radar, and were miscategorized, while 116 '0' loans were lumped in with the '1' loans based on the models predictions.



## Summary

  I had already began explaining how and for what reason the two models may excell in different areas -- the original being less likely to incorrectly identify a 'good loan'. Though one may argue that this would mean that the 'correct' model depends on the use-case, I would point to the confusion matrix results to identify that, though the second model is slightly more aggressive, it seems that it's ability to correctly identify the positive debtors was only reduced minimally -- incorrectly identifying only about 10 more, whereas its abilities in identifying risky debtors improved dramatically. The tradeoff is clearly a net-positive to almost all banks, as it decreases risk substantially in return for only minimal losses in overall profits. Furthermore, it seems that the best stategy to increasing overall profits is to find more debtors 'overall', in instead of allow the model to led more faulty debtors 'under the rug'. 

  For any financial institution, it is clear that the second mode, trained using the resampled data, would generate a net-positive for the firm in almost all circumstances. The numbers tell the story : out of around 18,765 'good debtors', the original model had incorrectly identified around 102 of them as 'faulty debtors' (who defaulted), whereas the resampled model had incorrectly identified even more -- 116. The tradeoff for this change of 14, out of a set of 18,765 positive debtors, is that the number of defaulting debtors who were incorrectly identified as 'positive debtors' had decreased from a value of 56, to only 4. In other tersm, the 'recall rate' (how many wouldn't get under the rug) improved from a value of 0.91, to a value of 0.99 (a 99% 'catch' rate). The numbers tell us that for a small loss of customers, we protect the firm from significant risk. 

  Finally, the fact that around 100 'positive loans' had remained incorrectly labelled as 'negative' would suggest that, though the debtors had actually payed the loans, the model understood that they were extremely unlikely to do so. The improvements in capabilities would actually showcase that those incorrectrly labelled individuals would also provide us information about what a good debtor looks like -- those 'on the fringe' will begin to default at a greater rate in a scenario where the economy takes a downturn. So, the resampled model has proved itself a much more efficient model for the firm in question. 




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
