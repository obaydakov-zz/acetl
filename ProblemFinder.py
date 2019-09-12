import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import os
from scipy import stats
import pprint
#import regex as re
from datetime import datetime
import unidecode

class ProblemFinder():
    def __init__(self, df):
        #* remove traiing spaces in column names
        df.columns=[col.strip() for col in df.columns]
        #* remove columns without names  - "Unnamed"
        drop_columns=[col for col in df.columns if col.find("Unnamed") >=0 ]
        df.drop(drop_columns, axis=1, inplace=True)
        
        #* clean spaces, convert to datetime object, assign NaN to empty strings 
        self.df=(df.apply(lambda col: pd.to_datetime(col, errors='ignore') 
                if (col.dtypes == object)
                else col, 
                axis=0)
                .apply(lambda col: pd.to_datetime(col, errors='ignore',unit='s') 
                if (col.dtypes == "int64" and col.mean()>1262304000 ) # 2010-01-01
                else col, 
                axis=0)
                .apply(lambda x: x.astype(str).str.upper().str.strip() if x.dtype == "object" else x)
                )
        
        
            
    #* get numeric columns
    def get_numeric_columns(self):
        list_columns= self.df._get_numeric_data().columns.tolist()
        return  [l for l in list_columns if len(l.strip())>0 ]

    #* missing values
    def missing_zero_values_table(self):
            df=self.df.replace('',np.nan)
            zero_val = (df == 0.00).astype(int).sum(axis=0)
            mis_val = df.isnull().sum()
            mis_val_percent = 100 * df.isnull().sum() / len(df)
            mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
            mz_table = mz_table.rename(
            columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
            mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
            mz_table['Data Audit'] = 100 * (1 - mz_table['Total Zero Missing Values'] / len(df))
            mz_table = mz_table[
                mz_table.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
            #print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            #    "There are " + str(mz_table.shape[0]) +
            #      " columns that have missing values.")
            return mz_table

    def calculate_missing_values_numeric(self):
        numeric_coluimns=self.get_numeric_columns()
        msv_table=self.missing_zero_values_table()
        return msv_table[msv_table["index"].isin(numeric_coluimns)][["index","Missing Values"]]
    
    #* identify columns with null values
    def columns_null_values(self):
        return self.df.columns[self.df.isnull().any()].values.tolist()

    #* identify columns with constant values 
    def columns_constant_vales(self):
        return self.df.columns[self.df.nunique()==1 ].values.tolist()

    #* identify ID columns
    def columns_ids(self):
        return [col  for col in  self.df.columns if len(self.df[col].unique().tolist())==self.df[col].shape[0] or col.lower().find("id")>=0 ]

    #* calculate number of duplicates in a column
    #* df = len(df) - df.nunique()
    def number_duplicates_column(self, col):
        df_pivot=self.df.pivot_table(index=[col], aggfunc='size')
        df_pivot = df_pivot[df_pivot > 1]
        return df_pivot.size
    
    #* Duplicate rows in data set
    def duplicates_in_dataset(self):
        counts=self.df.duplicated().value_counts()
        if True in counts.index:
            return counts.filter(items = [True]).values.flat[0]
        else:
            return 0
    #* Drop duplicates column
    
    
    
    #print(number_duplicates_column(df, 'Order Item Id'))

    #* intersection of two lsits
    @classmethod
    def unique_common_items(cls, list1, list2):
        # Produce the set of *unique* common items in two lists.
        return list(set(list1) & set(list2))
        #common= unique_common_items(col_need, columns_ids(df))

    #* return list of columns by type
    types=['floating','integer','bool','number','object','O']
    #* [dtype('int64') dtype('float64') dtype('O') dtype('<M8[ns]')]
    def columns_by_type(self, type_name):
        return self.df.select_dtypes(include=[type_name]).columns.values.tolist() 

    #* types -> column names dictionary
    def type_columns_dict(self):
        return {str(type): self.columns_by_type(type) for type in self.df.dtypes.unique()}

    #* identify currency columns:
    def columns_currency(self):
        return list(filter(lambda x: x != None,[ col 
                                       if True in set(self.df[col].str.contains(r'(\£|\$|\€)' , regex=True).unique()) 
                                       else None  
                                       for   col in self.df.columns if self.df[col].dtype == "object"] ))
    
    #* split currency column to two columns - vale and currency name
    def process_currency_column(self,col_name):
        self.df["Currency Name"]=self.df[col_name].str.extract('(\£|\$|\€)', expand=True)
        self.df[col_name] = self.df[col_name].replace(self.df["Currency Name"],'', regex = True)

    #* calculate number of year between today and datetime columns
    #* calculate outliers based  threshold
    def datetime_ouliers(self, df_dt_columns,threshold=2):
        #df = df_dt_columns.copy(deep=True)
        df = df_dt_columns
        for col in df.columns:
            df[col]=df[col].apply(lambda x: x.strftime('%Y-%m-%d'))
            
        enddt = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
        years={col:df[col].apply(lambda x: len(pd.date_range(start=x,end=enddt,freq='Y'))) for col in df.columns}
        outliers={col:{"Outliers":int(len(list(filter(lambda x: x >threshold , years[col]))))} for col in df.columns}
        return outliers