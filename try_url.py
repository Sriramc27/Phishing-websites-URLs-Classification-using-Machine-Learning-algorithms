# -*- coding: utf-8 -*-

#importing libraries
from sklearn.externals import joblib
import check_url

#load the pickle file
classifier = joblib.load('completed_models/svm_final.pkl')

#input url
print("enter url")
url = input()

#checking and predicting
checkprediction = check_url.main(url)
prediction = classifier.predict(checkprediction)
print(prediction)
