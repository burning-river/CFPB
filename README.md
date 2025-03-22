PREDICTING DISPUTED CONSUMER COMPLAINTS IN FINANCE
Final Project Report,
DATA*6300 - Analysis of Big Data, Winter 2024,
Saurabh Kumar (1281572),
MDS Student.

I INTRODUCTION: The Consumer Finance Protection Bureau (CFPB) is an independent agency of the United States government dedicated to providing protection to consumers from financial institutions [1]. The agency was created as a response to the financial crisis of 2007-08. Aggrieved consumers can submit complaints to the CFPB who redirects it to the responsible companies. Complaints are anonymized and published in the CFPB database when the company responds or after 15 days, whichever comes first. The consumer can then either accept or dispute the company response.

The database provides insights about the nature of complaints made by consumers in the financial sector. It identifies the most common concerns of past consumers thereby empowering people to make informed financial decisions. An annual report based on these complaints is presented to the US Congress enabling them to formulate new regulations on financial products to better serve customers.  

II PROBLEM STATEMENT: In this project, we aimed to predict whether a consumer will dispute a response from a financial institution from the information about the complaint published on the database. The attributes describing the complaints included: date when the complaint was submitted, company name,  financial product (mortgage, student loan, debt collection, etc.), type of issue (incorrect charges, loan servicing complaints, etc.), consumer complaint description, company response (both public and directed to consumer), customer location and time taken by company to respond. The output feature that we wanted to predict was whether the consumer disputed the company response to the complaint or not. This was a classification problem.

III POSSIBLE CONTRIBUTIONS: By predicting whether a consumer will dispute a company response, we tried to address the issues faced by both parties involved with the complaint: the consumer and the company. From a company standpoint, having disputed/unresolved complaints on a public database gives them a bad reputation. By predicting if consumers might dispute their response, the company can be more attentive towards such cases thereby preventing a future dispute. From feature importance analysis, we provided insights about attributes that contribute most towards a dispute and therefore recommend specific strategies to companies to satisfy consumers. From a prospective consumer perspective, predicting a disputed complaint can improve their decision making about buying financial products. From feature importance analysis, they can learn if they might experience an unresolved complaint in the future and steps to avoid being in such situations.     

IV METHODS: In this section, we describe the dataset in more detail. We discuss some pre-processing steps such as data cleaning, replacing missing values, and feature engineering techniques used to prepare our dataset before performing predictive modeling.
A DATASET: The dataset was downloaded from the following website [2]. The total number of records/complaints in the dataset is 555859 with 17 features in addition to the output feature - consumer_complained?. The output feature has class imbalance: 20% Yes vs. 80% No. The data was collected between the period of 11/30/2011 to 04/24/2016. The features contain timestamps and categorical values. Some features have missing values for example: state and zip 
code of consumers, description of consumer complaint, public response of companies, whether or not consent was provided by the consumer, etc.

Fig. 1: Class Distribution of the target variable

B DATA PRE-PROCESSING: In this section we describe the various data pre-processing steps prior to model building. Pre-processing the data was necessary in our case because there were columns with no useful information, missing values and categorical features in our dataset.
i. Data Cleaning: We first removed the column complaint_id since it contains no valuable information.
ii. Replacing Missing Values: Missing values were replaced on a case by case basis. Below is the breakdown of the columns containing missing values:

Fig.2: Barplot displaying the columns with missing values and the corresponding percentage of values missing.

For columns such as consumer_consent_provided, company_public_response, tags and consumer_complaint_narrative, close to or more than 80% values were missing at random. Since we have very few columns in our dataset, instead of dropping these columns, we replaced the missing values with value_unknown.
For sub_issue and sub_product, we imputed the missing values based on the issue and product values respectively. We found that all cases with the issues named ‘Loan modification, collection, foreclosure’ and ‘Loan servicing, payments, escrow account’ had missing sub_issue. This is perhaps because the issue itself is self-explanatory and doesn’t have sub-categories. Therefore for such cases, we replaced the missing sub_issue with the issue value. Similarly, all products named ‘Credit reporting’ and ‘Credit card’ had missing sub_products and therefore we replaced the missing values with the product value.
For zipcode, there were 77,470 incomplete entries with only the last 2 digits missing (for eg. ‘300XX’). For such cases, we looked at the first 3 digits of the zipcode and found the most frequent zipcode starting with those digits. For instance, the most frequent zipcode starting with ‘300’ was ‘30058’. Therefore, we replaced all values with ‘300XX’ with ‘30058’. There were also 4,407 completely missing zipcodes for which the state information was also missing. We therefore replaced such zipcodes with the most frequent zipcode in the most populous state in the US: ‘90046’ for California.
For missing state values, we replaced them with CA for California. 
iii Feature Engineering: We created the following new columns from existing ones because they provided more useful information. 
Time of Response: From the date of the complaint received by CFPB and the date sent by CFPB to the company, we created a new column capturing the time taken in days in sending the complaint to companies with the hypothesis being delays in sending complaints might lead to more disputes. 
Month and Year of Complaint: Since the exact date of the complaint sent may not be very useful, we made new columns containing the month and year of the complaints. The hypothesis was that certain periods in history might contain more disputes.
Latitude and Longitude: Using the pgeocode library in Python [3], we converted the zip codes of consumers to latitude and longitude. This was advantageous because there were 26,147 unique zip codes and one hot encoding would have created as many new columns. However, latitude and longitude values are continuous numerical values requiring only two new columns.  
Complaint description sentiment: The severity of the complaint might lead to it being disputed. Since the complaint description column contained a large amount of unstructured text, we used the pre-trained sentiment prediction model on hugging-face [4] to predict and create two new columns capturing the extent of negativity and neutrality of the description of the complaint.
iv Exploratory Data Analysis (EDA): We used PySpark to create pie charts and boxplots for EDA. We describe the key observations from EDA in the results section of the report.  
v Dummy Encoding: We used dummy encoding to convert all categorical features to numerical values. 
vi Normalization: The final pre-processing step was to normalize the numerical columns in our dataset using the Min-Max scaler. We used this method since the distribution of the numerical columns wasn’t Gaussian. After this, our dataset was ready to be trained on by machine learning classification models to predict consumer disputes of the complaints.   
C MODEL BUILDING: We first split the dataset with stratification to create a test set containing 10% rows from the entire dataset. For a justification on the choice of the split, please refer to the ‘Predictive Modeling’ section of the Python notebook. Due to class imbalance, we used random undersampling to balance the positive and negative examples in the training set. This allowed the classifier to be penalized for mis-classifying the class 1 examples as much as for class 0 examples. However, for the test set, we maintained the class imbalance. Following is the breakdown of the two classes in the training and test sets:


Class 0
Class 1
Training
100,680
100,680
Test
44,249
11,187







Table 1: Counts of class 0 and class 1 cases in training and test set.
Fig. 3: Class distribution of target variable in balanced train set and imbalance test set.

Our final training set was of size: 201,360 rows x 4019 columns. To handle such Big Data, we used a data compression technique known as Sparse Matrix. The use of sparse matrices in our dataset is also justified because most of our columns were obtained after dummy encoding and therefore contained mostly zeros. 

We then tested several classification models to see which ones outperform others in a reasonable amount of time. We tested Logistic Regression (for its simplicity and interpretability), Support Vector Classifier (for its ability to transform the data to a higher dimensional feature space which might create better separation between the two classes), Random Forest (since bagging models reduce variance) and XG Boost (since boosting models reduce bias). For each of these models, we tested different combinations of hyperparameters using grid search 5 fold cross-validation technique. We used the ROC AUC score to evaluate the models since it captures both the true and false positive rates for all probability threshold values between 0 to 1.

V RESULTS AND DISCUSSION: In this section, we describe the results from our analysis of the dataset. We will describe both the results from EDA in Spark and machine learning.
A RESULTS FROM EDA: Here are some top results from EDA along with supporting plots. For a more detailed discussion, please refer to our Python notebook where we have all the plots generated and mention every conclusion drawn from EDA in markdown.

Fig. 4: Pie chart showing top 5 companies in the dataset.


Fig. 5: Pie chart showing top 5 products in the dataset.

Fig. 6: Boxplot showing the distribution of response time by CFPB in the dataset.
	
Top results: 
Wells Fargo and Bank of America had the most complaints. Consumers can look at other banks to buy products. Better regulations in these companies can prevent consumer complaints. 
Citibank responses had higher than usual disputes from customers. 
Mortgage, debt collection and credit reporting were the products with most complaints. Both companies and consumers can be diligent while buying and selling such products respectively. 
Shorter response times by CFPB correlate with more disputes perhaps due to poorer pre-processing of the complaints by CFPB. This is something the CFPB could investigate.

B RESULTS FROM MACHINE LEARNING
From the 4 ML models that we tested, we obtained the following mean ROC AUC scores across 5 cross-validation folds:

Model
Score
Logistic Regression
0.636
Support Vector Classifier
0.635
Random Forest
0.624
XG Boost
0.645

Table 2: Performances of the 4 models selected in the project. 

We selected the XG Boost model for our final training since it had the best performance (effective) and low computational time (efficient). After doing a more granular grid search, we trained the fine tuned model on the entire balanced training set and tested the performance on the unbalanced test set. Below are the scores obtained from the test set (table 3), confusion matrix (table 4) and ROC AUC curve (Fig.7). We also extracted the top 10 features from the trained model (Fig. 8) and did a Mann Whitney U test to deduce categorical features that most distinguished between the two classes (Fig. 9).  

(True Label = 0, Predicted Label = 0)
22,332

(True Label = 0, Predicted Label = 1)
21,917

(True Label = 1, Predicted Label = 0)
3,349

(True Label = 1, Predicted Label = 1)
7,838


Metric
Score
Precision
0.26
Recall
0.70
F1-score
0.38
Accuracy
0.54
ROC AUC
0.64

						Table 4: Confusion matrix from the test set predictions.
Table 3: Various performance metrics of the final model  (probability threshold = 0.5). 

Fig. 7: ROC AUC curve from the test set.

Fig. 8: Top 10 features from predictive modeling.

Fig. 9: Observed and Expected distribution of company response.

Top results:
70% recall rate is a success since a large number of the complaints can be predicted which might be disputed in the future and therefore companies can take preemptive measures.
Overall, we could only modestly predict consumer disputes (ROC AUC = 0.64) and to improve the predictive model we might need better data: less missing values, more details such as monetary amount in the complaint, demographic information, etc.
Credit reporting, debt collection and money transfers were the products most important in our classification model. Both companies and consumers should be careful about such products.
Although disputed responses were closed with explanation more than expected, perhaps the explanation wasn’t satisfactory. An alternate explanation can be provided by companies.
We noticed that disputed cases were closed with monetary relief in lower numbers than expected. Companies can provide some monetary relief to satisfy consumers.

CONCLUSIONS: In this project we studied the publicly available dataset regarding consumer finance complaints filed in the US on the CFPB website. The dataset contained details about each complaint such as product, issue, location of the consumer, date of complaint, company response and consumer response in the form of either a complaint or agreement. The goal of the project was to classify the consumer response from the details of the complaint. The significance of this work is alerting companies regarding complaints that might be disputed in the future by consumers thereby harming their reputation. From the clients perspective, the project identifies products and companies they need to be careful about before purchasing.

The dataset was fairly Big as it contained about half a million rows and 17 columns. This was a classification problem on an unbalanced dataset (consumers disputed/class 1 = 20%, consumers agreed/class 0 = 80%). We began with performing EDA using Spark and identified Wells Fargo and Bank of America as top companies and mortgage and debt collection as the top products. We then pre-processed the data by removing unnecessary columns, filling in missing values and constructing new columns which were more useful than the ones in the raw dataset. After normalization and dummy encoding the categorical features, we then carved out 10% of the data as the test set and from the rest 90% created a balanced training dataset using random undersampling. We then compressed our data as a sparse matrix before testing Logistic Regression, Support Vector Classifier, Random Forest and XG Boost models. We selected the XG Boost model since it performed the best and fine tuned it. The performance of the final model on the test set was captured by the ROC AUC of 0.64 and Recall of 0.70. Even though the overall performance suggests that the predictive performance was modest, a recall of 70% can be considered a success since a large percentage of the disputed complaints can be correctly predicted by the model.
The dataset had some limitations: some features had more than 80% missing values and many important features were missing such as monetary amounts in the complaints. By identifying the top companies and products with the most complaints, we might be able to help both companies and consumers. The top companies should provide better service and consumers can go to other banks for similar products. Companies should also look at their current regulations regarding the top products to avoid future disputes. The company response also matters a lot. Current explanations provided by companies to settle complaints aren’t good enough and consumers do value some form of monetary or non-monetary relief or action from companies. CFPB should also investigate why shorter response time from their end leads to more disputes.

REFERENCES:
CFPB: https://www.consumerfinance.gov/
Dataset: https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints
Pgeocode: https://pypi.org/project/pgeocode/
Sentiment Analysis: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
