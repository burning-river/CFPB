# PREDICTING DISPUTED CONSUMER COMPLAINTS IN FINANCE

## Introduction: 
The Consumer Finance Protection Bureau (CFPB) is an independent agency of the United States government dedicated to providing protection to consumers from financial institutions [1]. The agency was created as a response to the financial crisis of 2007-08. Aggrieved consumers can submit complaints to the CFPB who redirects it to the responsible companies. Complaints are anonymized and published in the CFPB database when the company responds or after 15 days, whichever comes first. The consumer can then either accept or dispute the company response.

The database provides insights about the nature of complaints made by consumers in the financial sector. It identifies the most common concerns of past consumers thereby empowering people to make informed financial decisions. An annual report based on these complaints is presented to the US Congress enabling them to formulate new regulations on financial products to better serve customers.  

## Problem Statement: 
In this project, we aimed to predict whether a consumer will dispute a response from a financial institution from the information about the complaint published on the database. The attributes describing the complaints included: date when the complaint was submitted, company name,  financial product (mortgage, student loan, debt collection, etc.), type of issue (incorrect charges, loan servicing complaints, etc.), consumer complaint description, company response (both public and directed to consumer), customer location and time taken by company to respond. The output feature that we wanted to predict was whether the consumer disputed the company response to the complaint or not. This was a classification problem.

## Possible Contributions: 
By predicting whether a consumer will dispute a company response, we tried to address the issues faced by both parties involved with the complaint: the consumer and the company. From a company standpoint, having disputed/unresolved complaints on a public database gives them a bad reputation. By predicting if consumers might dispute their response, the company can be more attentive towards such cases thereby preventing a future dispute. From feature importance analysis, we provided insights about attributes that contribute most towards a dispute and therefore recommend specific strategies to companies to satisfy consumers. From a prospective consumer perspective, predicting a disputed complaint can improve their decision making about buying financial products. From feature importance analysis, they can learn if they might experience an unresolved complaint in the future and steps to avoid being in such situations.     

## Methods: 
### A DATASET: 
The dataset was downloaded from the following website [2]. The total number of records/complaints in the dataset is 555859 with 17 features in addition to the output feature - `consumer_complained?`. Overall, 1 out of 5 responses by companies to consumer complaints were disputed. The data was collected between the period of 11/30/2011 to 04/24/2016. The features contain timestamps and categorical values. Some features have missing values for example: 
* state and zip,
* code of consumers,
* description of consumer complaint,
* public response of companies,
* whether or not consent was provided by the consumer, etc.

Fig. 1: Class Distribution of the target variable

### Data pre-processing: 
We did thorough cleaning and pre-processing of the dataset. Missing and incomplete values were replaced with values inferred from other columns in the dataset or the most frequent values in the columns. For more details, please visit the Python notebook in the repo.

### Feature Engineering: 
We created the following new columns from existing ones because they provided more useful information. For example, we converted the zip codes of consumers to latitude and longitude. This was advantageous because there were about 26,000 unique zip codes and one hot encoding would have created as many new columns. However, latitude and longitude values are continuous numerical values requiring only two new columns. Similarly, the complaint description column contained a large amount of unstructured text. Therefore, we created two new columns capturing the extent of negativity and neutrality of the description of the complaint using sentiment analysis.

## Model Building: 
After feature engineering, we converted the categorical features into numerical form and normalized the dataset. We then split the dataset into training (80%), validation (10%), and test (10%) sets. We tested several training techniques such as imbalanced (weighted) and balanced learning (under/over sampling) to predict consumer disputes. We found that udnersampling was the best training method. We tested several classification models and found that the XGBoost model outperformed the others. For more details about this section, please visit the Python notebook.  

## Results and Discussion: 
* Wells Fargo and Bank of America had the most complaints. Consumers can look at other banks to buy products. Better regulations in these companies can prevent consumer complaints.
* Citibank responses had higher than usual disputes from customers. 

Fig. 4: Pie chart showing top 5 companies in the dataset.

* Mortgage, debt collection and credit reporting were the products with most complaints. Both companies and consumers can be diligent while buying and selling such products respectively. 

Fig. 5: Pie chart showing top 5 products in the dataset.

* Shorter response times by CFPB correlate with more disputes perhaps due to poorer pre-processing of the complaints by CFPB. This is something the CFPB could investigate.

Fig. 6: Boxplot showing the distribution of response time by CFPB in the dataset.

* 71% recall rate is useful since a large number of the complaints can be predicted which might be disputed in the future and therefore companies can take preemptive measures. To improve the predictive model we might need better data: less missing values, more details such as monetary amount in the complaint, demographic information, etc.
* Credit reporting, debt collection and money transfers were the products most important in our classification model. Both companies and consumers should be careful about such products.
* Company response is correlated with company disputes. Better explanation from companies, more monetary relief and alternative ways to compensate consumers can lead to less disputes.
  
REFERENCES:
CFPB: https://www.consumerfinance.gov/
Dataset: https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints
Pgeocode: https://pypi.org/project/pgeocode/
Sentiment Analysis: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
