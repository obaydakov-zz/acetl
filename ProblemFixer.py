import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import os
from scipy import stats
import pprint
#import regex as re
from datetime import datetime
import logging
import unicodedata


import warnings
warnings.filterwarnings('ignore')


class ProblemFixer():
    def __init__(self, df):
        #* remove traiing spaces in column names
        df.columns=[col.strip() for col in df.columns]
        #* remove columns without names  - "Unnamed"
        drop_columns=[col for col in df.columns if col.find("Unnamed") >=0 ]
        df.drop(drop_columns, axis=1, inplace=True)
        
        #* create logger
        self.logger=pd.DataFrame(columns=['time', 'message'])
        
        #* clean spaces, convert to datetime object, assign NaN to empty strings 
        self.df=(df.apply(lambda col: pd.to_datetime(col, errors='ignore',unit='s') 
                if (col.dtypes == object )  or (col.dtypes == "int64" and col.mean()>1262304000 ) # 2010-01-01
                else col, 
                axis=0)
                .apply(lambda x: x.astype(str).str.upper().str.strip() if x.dtype == "object" else x)
                )
        #* remove special characters from columns with type = 'object'
        #https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string
        [self.remove_special_characters(k) for k,v in self.df.dtypes.items() if v=="object"]
        
       
        
        #* Fill NaN based on user choice of the column
    def logger_append(self, time, level,message):
        self.logger=self.logger.append({'time': time, 'message': message}, ignore_index=True)
   
    def logger_save(self, file_name):
        self.logger.to_csv (file_name, index = None, header=True, sep='\t')
    
    def strip_accents(self,text):
        try:
            text = unicode(text, 'utf-8')
        except (TypeError, NameError): # unicode is a default on python 3 
            pass
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return str(text)
    
    def remove_special_characters(self,col):
        self.df[col]=self.df[col].apply(lambda x: self.strip_accents(x))
     
    #* process column with missing values   
    def missing_values_column(self,col, method,value=None):
        if method=="mean":
            self.df[col].fillna(self.df[col].mean(), inplace=True)
        if method=="median":
            self.df[col].fillna(self.df[col].median(), inplace=True)
        if method=="value" and not value:
            self.df[col].fillna(value)
        if method=="max":
            self.df[col].fillna(self.df[col].max(), inplace=True)
        if method=="mode":
            self.df[col].fillna(self.df[col].mode(), inplace=True)
        if method=="drop":
            self.df=self.df[self.df[col].notnull()]
        self.logger.info("PROCESS MISSING VALUES , COLUMN {}, METHOD {}".format(col, method))
    
    #* process column with negative values
    def negative_values_column(self, col, method, value=None):
        if method=="value" and not value:
            self.df[col][self.df[col] < 0] = value
        if method=="drop":
            #df.drop(df[df.score < 50].index, inplace=True)
            self.df = self.df[self.df[col] >= 0]
        if method=="abs":
            self.df[col] = self.df[col].abs()
        self.logger.info("PROCESS NEGATIVE VALUES , COLUMN {}, METHOD {}".format(col, method))
        
    #* drop rows with any missing values
    def missing_values_drop_rows(self):
        self.df.dropna(inplace=True)
        self.logger.info("DROP ROWS WITH MISSING VALUES")
        
    def split_date_time(self, col):
        self.df["{} Date".format(col)] = self.df[col].dt.date
        self.df["{} Time".format(col)] = self.df[col].dt.time
        #self.logger.info("SPLIT DATETIME COLUMN '{}' INTO '{}' AND '{}'".format(col, "{} Date".format(col),"{} Time".format(col)))
        self.logger_append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "INFO","SPLIT DATETIME COLUMN '{}' INTO '{}' AND '{}'".format(col, "{} Date".format(col),"{} Time".format(col)))
        #* remove time if it's constant
        if len(self.df["{} Time".format(col)].unique()) == 1:
            self.df.drop("{} Time".format(col),inplace=True,axis=1)
            #self.logger.info("REMOVE '{}' SINCE  IT'S CONSTANT".format("{} Time".format(col)))
            self.logger_append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "INFO","REMOVE '{}' SINCE  IT'S CONSTANT".format("{} Time".format(col)))
        #* split into hour, min, sec
        self.df[['h','m','s']] = pd.DataFrame([(x.hour, x.minute, x.second) for x in self.df["{} Time".format(col)]])
        #* remove if seconds are the same
        if len(self.df["s"].unique()) == 1:
            self.df["{} Time".format(col)]=self.df["h"].astype(str)+":"+self.df["m"].astype(str)
            #self.logger.info("REMOVE SECONDS FOR '{}' SINCE  IT'S CONSTANT".format("{} Time".format(col)))
            self.logger_append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "INFO","REMOVE SECONDS FOR '{}' SINCE  IT'S CONSTANT".format("{} Time".format(col)))
        self.df.drop(["h","m","s"],inplace=True,axis=1)    
            
    def extract_year_month_date(self,col):
        self.df["{} Year".format(col)] = pd.DatetimeIndex(self.df[col]).year
        self.df["{} Month".format(col)] = pd.DatetimeIndex(self.df[col]).month
        #self.logger.info("SPLIT '{}' INTO YEAR AND MONTH - '{}' , '{}' ".format(col, "{} Year".format(col),"{} Month".format(col) ))
        self.logger_append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "INFO","SPLIT '{}' INTO YEAR AND MONTH - '{}' , '{}' ".format(col, "{} Year".format(col),"{} Month".format(col) ) )

    def convert_epoch_datetime(self,col):
        #self.df["{} Date".format(col)] = pd.to_datetime(self.df[col], unit='s').dt.date
        #self.df["{} Time".format(col)] = pd.to_datetime(self.df[col], unit='s').dt.time
        self.df[col] = pd.to_datetime(self.df[col], unit='s')
        self.split_date_time(col)
        #self.logger.info("CONVERT '{}' DATE AND TIME - '{}' , '{}' ".format(col, "{} Date".format(col),"{} Time".format(col) ))
        self.logger_append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "INFO", "CONVERT '{}' DATE AND TIME - '{}' , '{}' ".format(col, "{} Date".format(col),"{} Time".format(col) ))

   
    def remove_rows_negative_values(self,col):
        self.df = self.df[self.df[col] >= 0]
        #self.logger.info("REMOVE ROWS WHERE   '{}' < 0".format(col))
        self.logger_append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "INFO","REMOVE ROWS WHERE   '{}' < 0".format(col))

    def remove_numerical_outlier_Z(self, col, z_thresh=1):
        normal = self.df[[col]].select_dtypes(include=[np.number]) \
                .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, result_type='broadcast').all(axis=1)
        # Drop (inplace) values set to be rejected
        outliers=(~normal).values.sum()
        self.df.drop(self.df.index[~normal], inplace=True)
        self.logger_append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "INFO", "REMOVE {} OUTLIERS FROM '{}' BASED ON Z CRITERIA, Z={} ".format(outliers,col,z_thresh))
        
    def remove_number_outliers_IRQ(self, col,low=0.25, high=0.75):
        Q1 = self.df[col].quantile(low)
        Q3 = self.df[col].quantile(high)
        IQR = Q3 - Q1
        normal = ~((self.df[col] < (Q1 - 1.5 * IQR)) |(self.df[col] > (Q3 + 1.5 * IQR)))
        outliers=(~normal).values.sum()
        # Keep only if True (not outliers)
        self.df=self.df[normal]
        self.logger_append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "INFO", "REMOVE {} OUTLIERS FROM '{}' BASED ON IRQ with low={} and high={} ".format(outliers, col,low,high))
    
    #* fill dates missing values uniformly
    def dates_missing_values(self,col):
        while True:
            if self.df[col].isnull().sum()==0:
                break 
        self.df[col].fillna(method="pad",limit=1, inplace=True)
        self.df[col].fillna(method="backfill",limit=1, inplace=True)
        