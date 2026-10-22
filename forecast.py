
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
#Load Dataset
df=pd.read_csv("Walmart.csv")
print("Dataset Shape:",df.shape)
print(df.dtypes)
print("Missing Values:\n",df.isna().sum())
#Convert Date and Sort Chronologically
df["Date"]=pd.to_datetime(df["Date"],format="%d-%m-%Y",errors="coerce")
df=df.dropna(subset=["Date","Weekly_Sales","Store"])
df=df.sort_values(["Store","Date"]).reset_index(drop=True)
#Extract Date Features
df["Year"]=df["Date"].dt.year
df["Quarter"]=df["Date"].dt.quarter
df["Month"]=df["Date"].dt.month
df["DayOfYear"]=df["Date"].dt.dayofyear
df["DayOfWeek"]=df["Date"].dt.dayofweek
#Create Lag Features Separately for Each Store
for lag in [1,4,52]:
    df[f"Sales_Lag_{lag}"]=df.groupby("Store")["Weekly_Sales"].shift(lag)
#Create Rolling Features Using Only Previous Weeks
for window in [4,12]:
    df[f"Rolling_Mean_{window}"]=df.groupby("Store")["Weekly_Sales"].transform(lambda x:x.shift(1).rolling(window=window).mean())
#Remove Rows Without Enough Historical Data
lag_cols=["Sales_Lag_1","Sales_Lag_4","Sales_Lag_52","Rolling_Mean_4","Rolling_Mean_12"]
df=df.dropna(subset=lag_cols).sort_values(["Date","Store"]).reset_index(drop=True)
print("Processed Dataset Shape:",df.shape)
#Chronological Train/Test Split Using Dates
unique_dates=np.sort(df["Date"].unique())
split_idx=int(len(unique_dates)*0.8)
cutoff_date=pd.Timestamp(unique_dates[split_idx])
train=df[df["Date"]<cutoff_date].copy()
test=df[df["Date"]>=cutoff_date].copy()
print("Training Period:",train["Date"].min(),"to",train["Date"].max())
print("Testing Period:",test["Date"].min(),"to",test["Date"].max())
print("Cutoff Date:",cutoff_date)
#Select Features and Target
features=["Store","Holiday_Flag","Temperature","Fuel_Price","CPI","Unemployment","Year","Quarter","Month","DayOfYear","DayOfWeek","Sales_Lag_1","Sales_Lag_4","Sales_Lag_52","Rolling_Mean_4","Rolling_Mean_12"]
X_train=train[features]
y_train=train["Weekly_Sales"]
X_test=test[features]
y_test=test["Weekly_Sales"]
#Encode Store as a Categorical Feature
preprocessor=ColumnTransformer(transformers=[("store",OneHotEncoder(handle_unknown="ignore"),["Store"])],remainder="passthrough")
#Build and Train Model
model=Pipeline(steps=[("preprocessor",preprocessor),("regressor",LinearRegression())])
model.fit(X_train,y_train)
#Make Predictions
pred=model.predict(X_test)
#Evaluate Model
mae=mean_absolute_error(y_test,pred)
mse=mean_squared_error(y_test,pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,pred)
print("\nModel Evaluation")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
#Permutation Feature Importance
importance=permutation_importance(model,X_test,y_test,n_repeats=10,random_state=42,scoring="neg_mean_absolute_error",n_jobs=-1)
feature_importance=pd.DataFrame({"Feature":features,"Importance":importance.importances_mean,"Std":importance.importances_std})
feature_importance=feature_importance.sort_values("Importance",ascending=False)
print("\nPermutation Feature Importance")
print(feature_importance.to_string(index=False))

#Save Predictions and Feature Importance
df.to_csv("walmart_sales_predictions.csv",index=False)
feature_importance.to_csv("walmart_feature_importance.csv",index=False)
print("\nSaved walmart_sales_predictions.csv")
print("Saved walmart_feature_importance.csv")
#Save Trained Model
import joblib
joblib.dump(model,"walmart_sales_model.joblib")
