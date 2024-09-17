import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle

def get_cleaned_data():
    df = pd.read_csv("data/breast_cancer.csv")
    # Drop NaN column and useless columns
    df.drop(["Unnamed: 32","id"],axis=1,inplace=True)
    # Replace B & M to 0 & 1
    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
    return df

def create_model(df):
    X = df.drop(["diagnosis"],axis=1,inplace=False)
    y = df["diagnosis"]
    # Normalise the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    # Fit the model
    model = LogisticRegression()
    model.fit(X_train,y_train)
    # Evaluation
    y_pred = model.predict(X_test)
    print("The accuracy score:{}".format(accuracy_score(y_test,y_pred)))
    print("The classification report:\n{}".format(classification_report(y_test,y_pred)))
    return model, scaler

def main():
    data = get_cleaned_data()

    model, scaler = create_model(data)

    #Export model and scaler as binary files using pickle
    #Reason not to use the file directly is because the model will be trained everytime the app is opened
    with open("model/model.pkl","wb") as f:   #wb means Writing in Binary 
        pickle.dump(model,f)
    with open("model/scaler.pkl","wb") as f:
        pickle.dump(scaler,f)

# The code wont be executed accidentally during importin, making the codes to be more robust
if __name__ == "__main__":
    main()