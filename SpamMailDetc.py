# Importing libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load the dataset to pandas data frame
dfMsgs = pd.read_csv("spamham.csv", encoding="ISO-8859-1")
dfMsgs.columns = ["category", "message", "extra1", "extra2", "extra3"]

# Adding non null extra columns as new rows
dfExtra0 = dfMsgs.loc[dfMsgs["message"].notnull(), ["category", "message"]]
dfExtra1 = dfMsgs.loc[dfMsgs["extra1"].notnull(), ["category", "extra1"]]
dfExtra2 = dfMsgs.loc[dfMsgs["extra2"].notnull(), ["category", "extra2"]]
dfExtra3 = dfMsgs.loc[dfMsgs["extra3"].notnull(), ["category", "extra3"]]

dfMsgs = pd.concat(
    [dfExtra0, dfExtra1.rename(columns={"extra1": "message"}), dfExtra2.rename(columns={"extra2": "message"}),
     dfExtra3.rename(columns={"extra3": "message"})], ignore_index=True)

# Lable spam mail as 0; non-spam (ham) mail as 1.
dfMsgs.loc[dfMsgs["category"] == "spam", "category",] = 0
dfMsgs.loc[dfMsgs["category"] == "ham", "category",] = 1

# Saperate the data as text and label. X --> text Y--> label
X = dfMsgs["message"]
Y = dfMsgs["category"]

# Split the data as train data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=3)

# Transform the text data to feature vector that can be used as input to SVM model using TfidfVectorizer.

# Convert the text to lower case.
ftrExtraction = TfidfVectorizer(min_df=1, lowercase="True")  # stop_words = "English"
X_trainFtr = ftrExtraction.fit_transform(X_train)
X_testFtr = ftrExtraction.transform(X_test)

# Convert Y_train and Y_test values into integers.
Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")

# Training the Support Vector Machine model with training data
model = LinearSVC()
model.fit(X_trainFtr, Y_train)

# PredTrainData = model.predict(X_trainFtr)
# AccrcyTrainData = accuracy_score(Y_train, PredTrainData)
#
# PredTestData = model.predict(X_testFtr)
# AccrcyTestData = accuracy_score(Y_test, PredTestData)

def detectSpam():
    pMsg = st.text_area("Enter any Message or Mail:")
    MsgFeatures = ftrExtraction.transform(pMsg)
    pRslt = model.predict(MsgFeatures)
    if pRslt[0] == 0:
        pRsltMsg = "It's SPAM!!!"
    else:
        pRsltMsg = "It's NOT SPAM!"
    return pRsltMsg


st.title("Spam Mail Detection")

st.text_input("Enter the message/mail you want to check..", "Type here...")
if (st.button("Check")):
    detectSpam()

st.success("It's NOT SPAM!")
st.exception("It's SPAM!!!")
