#!/usr/bin/env python
# coding: utf-8

# # Creating S3 Bucket

# In[39]:


import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker import image_uris
from sagemaker.session import s3_input, Session


# In[20]:


#creating bucket to begin the AWS E2E process
bucket_name = 'first-bank-application'

#
my_region = boto3.session.Session().region_name
print(my_region)


# In[21]:


#Accessing the s3 bucket
s3 = boto3.resource('s3')

#condition for creating the bucket
try:
    if my_region == 'us-east-2':
        s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint':'us-east-2'})
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ', e)


# In[22]:


# set an output path where the trained model will be saved, 
#Using the built-in Sagemaker xgboost algorithim
prefix = 'xgboost-as-a-built-in-algo'

#for storing model, will create a new folder, when a model is retrained
output_path = 's3://{}/{}/output'.format(bucket_name, prefix)
print(output_path)


# # Dowloading the Dataset and Storing in S3

# In[29]:


import pandas as pd
import numpy as np
import urllib
try:
    urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ',e)

try:
    model_data = pd.read_csv('./bank_clean.csv',index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)


# In[30]:


### Train Test split

train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)


# In[32]:


### Saving Train And Test Into Bucket format
## We start with Train Data
import os
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], 
                                                axis=1)], 
                                                axis=1).to_csv('train.csv', index=False, header=False)
##importing file to S3 Bucket
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
#creating path for for training data
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# In[36]:


# Test Data Into Buckets
pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')


# # Building and Training, sagemaker XGboost Algo 

# In[41]:


#Creating a container in the form of an image
container = image_uris.retrieve(region=boto3.Session().region_name, framework='xgboost', version="latest")


# In[42]:


#Hyperparameter tunning to adjust the algo
hyperparameters = {
    "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"binary:logistic",
        "num_round":50
}


# In[43]:


#creating a contructor called estimator to communicate with the XGBoost container image
estimator = sagemaker.estimator.Estimator(image_uri=container,
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          instance_type = 'ml.m5.2xlarge',
                                          instance_count= 1,
                                          volume_size = 5,
                                          output_path=output_path,
                                           use_spot_instances=True,
                                           max_run=300,
                                           max_wait= 600)


# In[44]:


#Fitting the model and calling the input train and test bucket within s3
estimator.fit({'train': s3_input_train, 'validation': s3_input_test})


# In[ ]:





# # Deploying Model to E2

# In[45]:


xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')


# # Prediction of the Test Data

# In[58]:


from sagemaker.serializers import CSVSerializer
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #Load the data into an array
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = CSVSerializer()
predictions = xgb_predictor.predict(test_data_array).decode('utf-8')
predictions_array = np.fromstring(predictions[1:], sep=',')
print(predictions_array.shape)


# In[60]:


cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))


# In[ ]:





# # Deleting Endpoints

# In[ ]:


#to prevent from continuous charging we should delete the project 
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()


# In[ ]:





# In[ ]:





# In[ ]:




