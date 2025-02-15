import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class Preprocessings:
    def __init__(self):
        self.df = {}

    def missing_values(self, df):
        # Display missing values and percentage of missing values per column
        missing_count = df.isna().sum()
        missing_percent = (missing_count / len(df)) * 100
        #print(f"Column '{col}': Missing Count = {missing_count}, Missing Percentage = {missing_percent:.2f}%")
            
        missing_data = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percent.map('{:.2f} %'.format)
        })

        print(missing_data)

        # Removing missing columns that have almost 100% and above 60% missings from the data 
        df = df.drop(['CrossBorder','WrittenOff','Rebuilt','Converted', 'NumberOfVehiclesInFleet', 'CustomValueEstimate', 'Bank'],axis=1)

    def handle_missing(self, df):
        df = df.copy()  # Create a copy to avoid modifying the original DataFrame

        for feature in df:
            if feature not in df.columns:
                print(f"Warning: '{feature}' column not found in DataFrame.")
                continue  # Skip this column and move to the next one

            if df[feature].isnull().all():  # Check if ALL values are NaN
                print(f"Warning: All values in '{feature}' column are missing. Backward fill will not work.")
                continue  # Skip this column and move to the next one

            df[feature] = df[feature].bfill()  # Backward fill missing values

        return df
    
    def feature_engineering(self, df):
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['RegistrationYear'] = pd.to_numeric(df['RegistrationYear'])

        # Create new features
        current_year = pd.Timestamp.now().year
        df['VehicleAge'] = current_year - df['RegistrationYear']

        # Calculate power-to-weight ratio
        df['PowerToWeightRatio'] = df['kilowatts'] / df['cubiccapacity']

        # Claim Frequency: Total claims over a specific period divided by the number of transactions.
        df['ClaimFrequency'] = df['TotalClaims'] / df['TransactionMonth'].dt.month
        df = df.drop(columns=['TransactionMonth', 'VehicleIntroDate'])  # Drop original date column if not needed

    def categorical_encoding(self, df):
        # One-Hot Encoding for features with more than two unique categories
        categorical_colums = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_colums, drop_first=True)

        # Label Encoding for binary features
        le = LabelEncoder()
        bool_cols = df.select_dtypes(include = 'bool').columns
        for cols in bool_cols:
            df[cols] = le.fit_transform(df[cols])
        return df
    
    def Train_Test_Split(self, df, test_size):
        # Define your target variable and features
        X = df.drop(columns=['TotalPremium', 'TotalClaims'])
        y = df['TotalClaims']  # or 'TotalPremium' based on your target variable

        # Split into 80% train and 20% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test 