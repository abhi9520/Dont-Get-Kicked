import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

train = pd.read_csv("C:/Users/tony/Desktop/car/training.csv")
test = pd.read_csv("C:/Users/tony/Desktop/car/test.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Quick data exploration to see what kind of data we are dealing with
train.describe()
train.dtypes
train.columns
train.shape
test.shape
test.describe()
test.dtypes
test.columns

#Graphing to see total number of missing data that we have. We have a total of 121690 rows for this dataset combined. Anything within this graph that has less than that shows how much missing data that specific column has.
#Here we can see that 'IsBadGuy' is missing which is fine since this is the variable that we are trying to predict, the missing data is coming from the test set for this column
#All other columns are not missing that many, so we can easily fix that by either taking the mean/mode imputation.
#Since 'PRIMEUNIT' and 'AUCGUART' is missing more than 95% of data, we might as well remove them because they would not add any value in our prediction.
ms.bar(train)
ms.bar(test)

#--------------------------------------------------------------------------------------------------
#REMOVE OUTLIERS BASED ON BOXPLOT
#--------------------------------------------------------------------------------------------------
train.boxplot()


train = train[(train['VehOdo'] < 112000) & (train['VehOdo'] > 5000) & (train['MMRAcquisitionAuctionAveragePrice'] < 17400) & (train['MMRAcquisitionAuctionCleanPrice'] < 20000) &
                            (train['MMRAcquisitonRetailCleanPrice'] > 1) & (train['MMRAcquisitonRetailCleanPrice'] < 25000) & (train['MMRCurrentAuctionAveragePrice'] < 17500) & (train['MMRCurrentAuctionAveragePrice'] > 1) &
                            (train['VehBCost'] > 1700) & (train['VehBCost'] < 12500) & (train['WarrantyCost'] < 2600) & (train['VehOdo'] > 15000)]

#--------------------------------------------------------------------------------------------------
#TRAINING SET/HOLD-OUT DATA CLEANING
#--------------------------------------------------------------------------------------------------

#Filling in missing object variables with the mode
for range in ['Trim', 'SubModel', 'Color', 'Transmission', 'WheelTypeID', 'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName']:
    train[range] = train[range].fillna(train[range].mode()[0])

#Replacing missing value for numeric variables with the mean
for range1 in ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',  'MMRCurrentRetailCleanPrice']:
    train[range1] = train[range1].fillna(train[range1].mean())

#--------------------------------------------------------------------------------------------------
#TEST SET DATA CLEANING
#--------------------------------------------------------------------------------------------------
for range in ['Trim', 'SubModel', 'Color', 'Transmission', 'WheelTypeID', 'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName']:
    test[range] = test[range].fillna(test[range].mode()[0])


#Replacing missing value for numeric variables with the mean
for range1 in ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',  'MMRCurrentRetailCleanPrice']:
    test[range1] = test[range1].fillna(test[range1].mean())

#--------------------------------------------------------------------------------------------------
#DATA PREPROCESSING/FEATURE ENGINEERING/EXTRACTING
#--------------------------------------------------------------------------------------------------

#Combining both train and test together to clean data at the same time
combined = train.append(test, ignore_index= True, sort = False)

#Dropping 'PRIMEUNIT' and 'AUCGUART' as 95%+ of the data is missing
combined.drop(combined.columns[26:28], axis = 1, inplace = True)

#Color NOT AVAIL can be converted to OTHER
combined["Color"].replace("NOT AVAIL", "OTHER", inplace = True)


#Data Transformation/Feature Engineering for PurchDate, splitting into two variables Months and Year
combined['PurchDate'] = pd.to_datetime(combined['PurchDate'])
combined['Purch_Month'] = combined['PurchDate'].dt.strftime('%b')
combined['Purch_Year'] = combined['PurchDate'].dt.year


combined['LikelyBrokenAverageCars'] = np.where(combined['MMRAcquisitionAuctionAveragePrice'] < combined['MMRCurrentAuctionAveragePrice'], "Likely", "Not")
combined['LikelyBrokenGoodCars'] = np.where(combined['MMRAcquisitionAuctionCleanPrice'] < combined['MMRCurrentAuctionCleanPrice'], "Likely", "Not")

#Loop through columns to change data type to categorical
for col in ['Make', 'Model', 'Trim', 'SubModel', 'Color', 'Transmission', 'WheelTypeID', 'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName', 'IsOnlineSale', 'VNST', 'Auction', 'Purch_Month', 'Purch_Year', 'VehicleAge', 'LikelyBrokenAverageCars', 'LikelyBrokenGoodCars']:
    combined[col] = combined[col].astype('category')


#Relooking at the data, there were some redundant data that can be removed, such as VehYear which is already described by the VehicleAge column.
#WheelTypeID which can be removed as we have the WheelType. VNZIP1 can also be removed as we have the states for each vehicle in VNST.
#I also removed RefID as it's just an identifier and would not add value in our prediction.
#Removed Purch_Date as well since we performed feature engineering based on this column.
#BYRNO did not seem signnificant to me as it should not determine whether a car is kicked or not based on who bought the vehicle, so I removed it.
combined.drop(combined.columns[[0,2, 4,12, -9, -10,8]], axis = 1, inplace = True)

combined.dtypes

#Splitting data
training_set = combined[0:70479]
testing_set = combined[70479:]

#--------------------------------------------------------------------------------------------------
#DATA EXPLORATION/VISUALS
#--------------------------------------------------------------------------------------------------
training_set.hist(bins= 50, figsize = (20,15))

training_set['IsBadBuy'] = training_set['IsBadBuy'].astype('category')

features = ['VehBCost', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
            'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']

scatter_matrix(training_set[features], figsize = (20, 15))

#--------------------------------------------------------------------------------------------------
#FEATURE SCALING - TRAINING
#--------------------------------------------------------------------------------------------------

training_label = training_set['IsBadBuy']

#Removing variable to scale data
training_scale = training_set[['MMRAcquisitionAuctionAveragePrice', 'VehBCost', 'VehOdo', 'WarrantyCost', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',  'MMRCurrentRetailCleanPrice']]
training_set.drop(training_set[['MMRAcquisitionAuctionAveragePrice', 'VehBCost', 'VehOdo', 'WarrantyCost', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',  'MMRCurrentRetailCleanPrice', 'IsBadBuy']], axis = 1, inplace = True)

#Scaling
scale = StandardScaler()
scaled_data = scale.fit_transform(training_scale)
scale = pd.DataFrame(data = scaled_data, columns = training_scale.columns)

#Combining data
training_final = pd.concat([training_set, scale], axis = 1, sort = False)
#--------------------------------------------------------------------------------------------------
#FEATURE SCALING - TESTING
#--------------------------------------------------------------------------------------------------

#Removing variable to scale data
testing_scale = testing_set[['MMRAcquisitionAuctionAveragePrice', 'VehBCost', 'VehOdo', 'WarrantyCost', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',  'MMRCurrentRetailCleanPrice']]
testing_set.drop(testing_set[['MMRAcquisitionAuctionAveragePrice', 'VehBCost', 'VehOdo', 'WarrantyCost', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',  'MMRCurrentRetailCleanPrice', 'IsBadBuy']], axis = 1, inplace = True)

#Scaling
scale = StandardScaler()
scaled_test_data = scale.fit_transform(testing_scale)
scale_test = pd.DataFrame(data = scaled_test_data, columns = testing_scale.columns)

#Combining data
testing_final = pd.concat([testing_set.reset_index(), scale_test.reset_index()], axis = 1, sort = False, ignore_index= False)
testing_final.drop(testing_final.columns[[0, 18]], axis = 1, inplace = True)
testing_final.dtypes
#--------------------------------------------------------------------------------------------------
#ONE HOT ENCODING - Creating Dummy Variables for categorical data
#--------------------------------------------------------------------------------------------------
#combined_for_dummies = training_final.append(testing_final, ignore_index= True, sort = False)

#Changing all categorical variables into dummies using One Hot Encoding
#combined_final = pd.get_dummies(combined_for_dummies, sparse= False)

----------------------------------------------------------------------------------
#MODEL DEVELOPMENT
#--------------------------------------------------------------------------------------------------
training_final = pd.concat([training_final, training_label], axis = 1, sort = False)

training_final.dtypes
testing_final.dtypes
training_final.to_csv('training_v3.csv')
testing_final.to_csv('testing_v3.csv')

