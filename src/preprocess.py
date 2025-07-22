import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocessing(path=r'C:\Users\kavyasree\Desktop\Career_counciller\data\student-scores.csv'):
    df=pd.read_csv(path)
    #dropping name as it is not requirred
    df.drop(['id','first_name','last_name','email'],axis=1,inplace=True)# axis 1= col and inplace is used to modify orginal data
    encoder={}
    label_encoder=['gender','part_time_job','extracurricular_activities','career_aspiration']
    
    for col in label_encoder:
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col].astype(str))
        encoder[col]=le
    X=df.drop(['career_aspiration'],axis=1)#features
    y=df['career_aspiration']#target to predict
    return X,y,encoder


#testing
X, y, encoders = preprocessing()
print("✅ X shape:", X.shape)
print("✅ Sample data:\n", X.head())
print(y.head())
