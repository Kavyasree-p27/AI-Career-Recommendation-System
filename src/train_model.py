import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib  #used to save the trained model
from sklearn.metrics import classification_report
from preprocess import preprocessing

X,y,encoder=preprocessing()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("classification report:\n")
print("prediction rate: %d",y_pred)
print(classification_report(y_test,y_pred))

joblib.dump(model, 'models/career_model.pkl')#saved the trained model
joblib.dump(encoder, 'models/encoders.pkl')#saves the list of labeled encoder

print("model loading completed.")
