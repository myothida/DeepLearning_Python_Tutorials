import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DataExplorer:
    def __init__(self, df):
        """
        Initializes the DataExplorer class with a given DataFrame.

        Parameters:
        df (pandas.DataFrame): The dataset to be explored.
        """
        self.df = df

    def describe_data(self):
        """
        Return a DataFrame with two columns: column names and data types.

        Returns:
        pandas.DataFrame: A DataFrame with column names and their respective data types.
        """        
        data_description = pd.DataFrame({
            'Column Name': self.df.columns,
            'Data Type': self.df.dtypes
        }).reset_index(drop=True)
        return data_description
        
    def check_missing_values(self):
        """
        Check for missing values in the dataset and return a summary.

        Returns:
        pandas.DataFrame: A DataFrame showing the number of missing values per column.
        """
        missing_values = self.df.isnull().sum()
        return missing_values
    
    def check_linear_correlation(self, feature, target):
        """
        Check if a feature is correlated to the target.

        Returns:
        Linear corrlation score
        """
        cor_scores = self.df[[feature, target]].corr()
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        regression_types = [
            {"title": "Linear Regression", "order": 1, "logx": False},
            {"title": "Non-linear Regression (Order = 2)", "order": 2, "logx": False},
            {"title": "Log-Linear Regression", "order": None, "logx": True},
        ]

        for ax, reg_type in zip(axes, regression_types):
            if(reg_type["logx"]==True):
                sns.regplot(x=feature, y=target,data=self.df,ax=ax,color='k', marker = '.',  logx=reg_type["logx"])
            else:
                sns.regplot(x=feature, y=target,data=self.df,ax=ax,color='k', marker = '.', order=reg_type["order"])
            ax.set_title(reg_type["title"])
            ax.set_xlabel(feature)
            ax.set_ylabel(target)

        plt.tight_layout()
        plt.show()
        return cor_scores
    

    def check_outliers(self, feature):
        """
        Check for outliers in a given feature both numerically and graphically.

        Parameters:
        - feature: The feature column name
        """
        # Numerical Check using IQR
        Q1 = self.df[feature].quantile(0.25)
        Q3 = self.df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find the outliers
        outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
        # Graphical Check using Boxplot
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=self.df[feature], color='lightgreen')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.show()

        if(outliers.shape[0]>1):
            print(f"Numerically Identified Outliers for {feature}:")
            return outliers
        else:
            return None

 
