# Tanzanian Wells - Classification Modeling

<p align="center">
  <img src="images/tanzania-water-well-3.jpg" />
</p>

## 1. Overview

This notebook examines Tanzania's water wells, and uses classification models to predict whether a water point is non-fonctional.
The organization of this notebook follows the CRoss Industry Standard Process for Data Mining (CRISP-DM) is a process model that serves as the base for a data science process.


## 2. Business Understanding

This notebook examines Tanzania's water wells, and uses classification models to predict whether a water point is non-fonctional.
The organization of this notebook follows the CRoss Industry Standard Process for Data Mining (CRISP-DM) is a process model that serves as the base for a data science process.

## 3. Data Understanding

The data comes from drivendata.org, a platform which hosts data science competitions with a focus on social impact. The source of data provided by DrivenData is the Tanzanian Ministry of Water, and is stored by Taarifa. 

The actual dataset can be found [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) under the 'Data download section'. 

4 files are indicated. The below files were downloaded and renamed as follows:
- Training set values: training_set_values
- Training set labels: training_set_labels
- Test set values: test_set_values

These are the files used for the main modeling and predictive analysis. 
<br>
The test set values file is the one used to measure the accuracy of the model.

The data contains 59,400 rows and 39 feature columns with 1 id column. Here is the description of each column:

* `amount_tsh`: Total static head (amount water available to waterpoint)
* `date_recorded`: The date the row was entered
* `funder`: Who funded the well
* `gps_height`: Altitude of the well
* `installer`: Organization that installed the well
* `longitude`: GPS coordinate
* `latitude`: GPS coordinate
* `wpt_name`: Name of the waterpoint if there is one
* `num_private`: No description was provided for this feature
* `basin`: Geographic water basin
* `subvillage`: Geographic location
* `region`: Geographic location
* `region_code`: Geographic location (coded)
* `district_code`: Geographic location (coded)
* `lga`: Geographic location
* `ward`: Geographic location
* `population`: Population around the well
* `public_meeting`: True/False
* `recorded_by`: Group entering this row of data
* `scheme_management`: Who operates the waterpoint
* `scheme_name`: Who operates the waterpoint
* `permit`: If the waterpoint is permitted
* `construction_year`: Year the waterpoint was constructed
* `extraction_type`: The kind of extraction the waterpoint uses
* `extraction_type_group`: The kind of extraction the waterpoint uses
* `extraction_type_class`: The kind of extraction the waterpoint uses
* `management`: How the waterpoint is managed
* `management_group`: How the waterpoint is managed
* `payment`: What the water costs
* `payment_type`: What the water costs
* `water_quality`: The quality of the water
* `quality_group`: The quality of the water
* `quantity`: The quantity of water
* `quantity_group`: The quantity of water
* `source`: The source of the water
* `source_type`: The source of the water
* `source_class`: The source of the water
* `waterpoint_type`: The kind of waterpoint
* `waterpoint_type_group`: The kind of waterpoint


## 4. Data Preparation

### 4. a. Joining values and labels datasets together
The first step of preparing the data is to merge both df_values and df_labels, as the latter contains the target value.
Both datasets are merged on the 'id' column.

### 4. b. Data transformation & cleaning

In this section, missing values were handled, unnecessary columns were removed based on context and to avoid duplicates. Categorized features were further grouped following research and finally transformed through one-hot encoding. 

An initial exploration was done by visualizing box plots of a set of columns, which led to the conclusion that scaling would be required.


The below  columns were removed for the following reasons:

1. Irrelevant for predictions (i.e. date the row was entered, waterpoint name)
2. Contains similar information as another column (i.e. extraction_type, water_quality) 
3. Contains information which would require additional conversion (i.e. region_code, district_code)

* `id`: the identification number assigned to the water well 
* `date_recorded`: The date the row was entered
* `longitude`: GPS coordinate
* `latitude`: GPS coordinate
* `wpt_name`: Name of the waterpoint if there is one
* `num_private`: undefined
* `subvillage`: Geographic location
* `region_code`: Geographic location (coded)
* `region`: Geographic location. There are 21 regions, while location by basin can be provided with 9 categories. Choosing less detailed categories is preferred to prevent creating a sparse dataframe 
* `district_code`: Geographic location (coded)
* `lga`: Geographic location
* `ward`: Geographic location
* `recorded_by`: Group entering this row of data
* `scheme_management`: Who operates the waterpoint
* `extraction_type`: The kind of extraction the waterpoint uses
* `extraction_type_group`: The kind of extraction the waterpoint uses
* `management_group`: How the waterpoint is managed
* `payment`: What the water costs
* `payment_type`: Frequency of payment: while it would be interesting to understand link between payment and well functionality, this feature has no link with the quality of water type and should be investigated separately
* `water_quality`: The quality of the water
* `quantity_group`: The quantity of water
* `source`: The source of the water
* `source_class`: The source of the water
* `waterpoint_type`: The kind of waterpoint
* `waterpoint_type_group`: provides similar information as source type and extraction type


## 5. Modeling
For each model, the same structure was followed: 
  1. A train-test split was perfromed 
  2. A baseline model was built and evaluated 
  3. Data preprocessing techniques were applied if necessary - particularly for the first model type 
  4. Additional models were built, where parameters were tuned
  5. The models were evaluated using the chosen classification metrics for this problem: recall, accuracy and log loss.  
  6. A final model was chosen  


5. a. Logistic Regression 

Logistic Regression was the first model. As a consequence, more pre-processing techniques were applied:
  * Class Imbalance was addressed using SMOTE for the minority class 'non-functional' 
  * Stratified K-Fold cross validation technique was also applied to build a more robust model despite class imbalance
  * Hyperparameters were tuned: regularization was reduced and an alternative solver was applied.
  * Finally, the best number of features (45) was chosen thanks to the application of Recursive Feature Elimination technique. 

The best model was chosen from the results provided by the Recursive Feature Elimination technique.

Results for this model were as follows: 

<u>Log Loss</u>: 0.5437699554671055
The log loss is below 1 and indicate a decent accuracy in predicting probabilities and consequently, a decent model performance.

<u>Recall Score</u>: 0.4866617538688283
The true positive rate metric measuring the proportion of actual positive cases that the model correctly identifies is low. Less than 50% of positive cases were correctly identified.

<u>Accuracy Score</u>: 0.7140740740740741

The overall correctness of the model's predictions, considering both true positives and true negatives is decent: approximatively 71.41.%

--------ADD Results-------- 

![](images/RFE_models.png)

5. b. K-Nearest Neighbors 

K-Nearest Neighbors model was the second one used to make predictions. As a consequence, more pre-processing techniques were applied:
  * Class Imbalance was addressed using SMOTE for the minority class 'non-functional' 
  * Stratified K-Fold cross validation technique was also applied to build a more robust model despite class imbalance
  * Hyperparameters were tuned: regularization was reduced and an alternative solver was applied.
  * Finally, the best number of features (45) was chosen thanks to the application of Recursive Feature Elimination technique. 

The best model was chosen from the results provided by the Recursive Feature Elimination technique.

![](images/RFE_models.png)




## 6. Evaluation



## 7. Findings & Recommendations 



## 8. Limits & Next Steps


## For More Information 
See the full analysis and code in the [Jupyter Notebook](notebook.pdf) as well as summary in this [presentation](presentation.pdf).


For additional info, contact [Albane Colmenares](mailto:albane.colmenares@gmail.com?subject=[GitHub]%20Source%20Han%20Sans)

## Repository Structure
```
├── images
├── .gitignore
├── README.md
├── tanzanian-wells.ipynb
├── presentation.pdf
└── notebook.pdf

```
