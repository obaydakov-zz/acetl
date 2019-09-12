#https://github.com/Miserlou/lambda-packages
#https://chrisalbon.com/python/data_wrangling/pandas_regex_to_create_columns/
import json
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import os
from scipy import stats
import pprint

#import regex as re
from datetime import datetime

from ProblemFinder import ProblemFinder
from ProblemFixer import ProblemFixer

pd.set_option('display.max_columns', 20)
#from typing import List

col_need=["Order Item Id" ,"Order Type" ,"Lazada Id"   ,"Seller SKU","Lazada SKU","Created at","Updated at"]
col_optional=[]

data_path = 'Data//retail_DS.csv'
data_path = 'Data//orderDataFlat.csv'
#missing_values = ["n/a", "na", "--"]
#na_values = missing_values
df = (pd.read_csv(data_path))
#print(df.head())

git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch read_s3.py" \
  --prune-empty --tag-name-filter cat -- --all

#accented_string = "CARROT ËêùÂçú"
# accented_string is of type 'unicode'
#import unidecode
#print(unidecode.unidecode(accented_string))

    
#* try to convert column value to int and calculate errors
def calculate_conversion_errors(df_column):
    cnt=0
    errors=[]
    for row in df_column:
        try:
            int(row)
        except ValueError:
            print(row)
            errors.append(cnt)
            pass
        cnt+=1
    return errors

#* Try to remove letters and keep numbers 
def remove_letters_from_column(df_column):
    return pd.to_numeric(df_column.astype(str).str.replace(r'\D', '') , errors='ignore')

#* Calculate negative values for numeric columns
def calculate_negative_values(df_column):
    return df_column.lt(0).sum()

def calculate_numerical_outliers_Z(df, col_name, z_thresh=1):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df[[col_name]].select_dtypes(include=[np.number]) \
                .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, result_type='broadcast').all(axis=1)
    # Drop (inplace) values set to be rejected
    #df.drop(df.index[~constrains], inplace=True)
    return constrains[constrains==False].shape[0]
    #return df[[col_name]].shape[0] - df[[col_name]].index[constrains].shape[0]

def calculate_number_outliers_IRQ(df, col_name,low=0.25, high=0.75):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col_name] < (Q1 - 1.5 * IQR)) |(df[col_name] > (Q3 + 1.5 * IQR)))        
    return df[outliers].shape[0]

def Analysys(path, filename):
    print(path)
    df = pd.read_csv(path)
    A=ProblemFinder(df)
    
    id_columns=A.columns_ids()
    constant_columns=A.columns_constant_vales()
    currency_columns=A.columns_currency()
    msv_table_all=A.missing_zero_values_table()
    mcv_columns=list(msv_table_all.index)


    #* Check numeric columns
    numeric_columns=A.get_numeric_columns()
    if numeric_columns:
        #msv_table=msv_table_all.loc[numeric_columns,"Missing Values"].fillna(0)

        msv_dict={num_column:{"Missing values":int(msv_table_all.loc[num_column,"Missing Values"])  
                            if num_column in mcv_columns else 0} 
                for num_column in numeric_columns}
        msv_percent_dict={num_column:{"Data Audit, %":int(msv_table_all.loc[num_column,"Data Audit"])  
                            if num_column in mcv_columns else 0} 
                for num_column in numeric_columns }
        #print(msv_percent_dict)
            
        negv_dict={num_column:{"Negative values":int(calculate_negative_values(A.df[num_column]))}  
                for num_column in numeric_columns}
        
        outliers_dict={num_column:{"Outliers":int(calculate_numerical_outliers_Z(A.df,num_column)) 
                                   if (num_column not in constant_columns and num_column not in id_columns) else 0}
                    for num_column in numeric_columns }
        
        id_dict={ num_column:{"ID column":True if num_column in id_columns else False} 
                for num_column in numeric_columns  }
        const_dict={ num_column:{"Constant column":True if num_column in constant_columns else False} 
                    for num_column in numeric_columns  }
        duplicates_dict={num_column:{"Number of duplicates":int(A.number_duplicates_column(num_column))} 
                        for num_column in numeric_columns}
        currency_dict={ num_column:{"Currency column":True if num_column in currency_columns else False} 
                    for num_column in numeric_columns  }
        type_dict={num_column:{"Type":"Numeric"} for num_column in numeric_columns} 
        final_numeric={key: {**value,**negv_dict[key], **msv_percent_dict[key],
                        **outliers_dict[key],**id_dict[key], 
                        **const_dict[key],**duplicates_dict[key],
                        **currency_dict[key]
                        } for key, value in msv_dict.items()}
        #pprint.pprint(final_numeric)

    #* check "object" columns
    object_columns=A.type_columns_dict().get("object",None)
    if object_columns:
        #msv_table=msv_table_all.loc[object_columns,"Missing Values"].fillna(0)
        msv_dict={num_column:{"Missing values":int(msv_table_all.loc[num_column,"Missing Values"])
                            if num_column in mcv_columns else 0} 
                for num_column in object_columns}
        
        msv_percent_dict_string={num_column:{"Data Audit, %":int(msv_table_all.loc[num_column,"Data Audit"])  
                            if num_column in mcv_columns else 0} 
                for num_column in object_columns }
                
        id_dict={ num_column:{"ID column":True if num_column in id_columns else False} 
            for num_column in object_columns  }
        const_dict={ num_column:{"Constant column":True if num_column in constant_columns else False} 
                for num_column in object_columns  }
        duplicates_dict={num_column:{"Number of duplicates":int(A.number_duplicates_column(num_column))} 
                    for num_column in object_columns}
        currency_dict={ num_column:{"Currency column":True if num_column in currency_columns else False} 
                for num_column in object_columns  }
        type_dict={num_column:{"Type":"String"} for num_column in object_columns} 
        final_object ={key: {**value, **id_dict[key], 
                        **const_dict[key],**duplicates_dict[key],
                        **currency_dict[key],
                        **msv_percent_dict_string[key],
                        **type_dict[key]

                        } for key, value in msv_dict.items()}
        #pprint.pprint(final_object)

    #* check datetype columns
    datetime_columns=A.type_columns_dict().get("datetime64[ns]",None)

    #print(datetime_columns)
    #print(msv_table_all.index)
    #print(mcv_columns)
    if datetime_columns:
        msv_dict={dt_column:{"Missing values":int(msv_table_all.loc[dt_column,"Missing Values"]) if dt_column in mcv_columns else 0 } 
                for dt_column in datetime_columns}
        id_dict={ dt_column:{"ID column":True if dt_column in id_columns else False} 
            for dt_column in datetime_columns  }
        const_dict={ dt_column:{"Constant column":True if dt_column in constant_columns else False} 
                for dt_column in datetime_columns  } 
        #outliers_dict= A.datetime_ouliers(A.df[datetime_columns],2)
        
        msv_percent_dict_dt={dt_column:{"Data Audit, %":int(msv_table_all.loc[dt_column,"Data Audit"])  
                            if dt_column in mcv_columns else 0} 
                for dt_column in datetime_columns }
        
        type_dict={dt_column:{"Type":"Datetime"} for dt_column in datetime_columns} 
        final_datetime ={key: {**value, **id_dict[key], 
                        **const_dict[key],
        #                **outliers_dict[key],
                        **msv_percent_dict_dt[key],        
                        **type_dict[key]
                        } for key, value in msv_dict.items()}


    print(A.df.shape[1])
    print(len(final_numeric.keys())+len(final_object.keys())+len(final_datetime.keys()))
    all_col=A.df.columns
    processed_col=list(final_numeric.keys())+list(final_object.keys())+list(final_datetime.keys())
    print(list(set(all_col) - set(processed_col)))
    final_all={**final_numeric,**final_object,**final_datetime}
    
    final=dict()
    final["file_name"]="{}_issue_report.json".format(filename)
    final["Row duplicates"]=A.duplicates_in_dataset()
    final["Total rows"]=A.df.shape[0]
    final["Total columns"]=A.df.shape[1]
    final["Columns"]=final_all
    with open("issue_reports\\"+final["file_name"], 'w') as outfile:
        json.dump(final, outfile)
    print("DONE")


df1=pd.DataFrame({'date': [1546315200,	1545553318,	1545553318,"2019/09/08"],
                 'date_new': ["2019/09/08",	np.nan,	"2019/09/10","2019/09/11"],
                 "string":["BEETROOT������ʆ?","CARROT ËêùÂçú","CARROT ËêùÂçú","CARROT ËêùÂçú"],
                 "negative":[0,  3, -9, -90]})

data_path = 'Data//orderDataFlat.csv'

import glob
path = "Data//*.csv"
for fname in glob.glob(path):
    print(fname)
    Analysys(fname.replace("\\","//"), fname.split("\\")[1].split(".")[0])

'''
df = (pd.read_csv(data_path))
print(df.columns)
print(len(df['DataMessageGUID                      '].unique().tolist()))
print(df['DataMessageGUID                      '].shape[0])
df_pivot=df.pivot_table(index=['DataMessageGUID                      '], aggfunc='size')
print(df_pivot[df_pivot > 1].head())
'''

#A=ProblemFinder(df)
#print(A.columns_ids())

#P=ProblemFixer(df)
#print(P.df.head())

#for col in df.columns:
#    if (col.dtypes == object )  or (col.dtypes == "int64" and df[col].mean()>1262304000 ):
#        df[col]=pd.to_datetime(df[col], errors='ignore',unit='s')
#print(df.head())

#print(calculate_number_outliers_IRQ(df, "negative"))

#df["date_new"]=df["date_new"].interpolate(method='time', limit=1)
#print(type(df.dtypes))
#[print(k) for k,v in df.dtypes.items() if v=="object"]

#df = (pd.read_csv(data_path))
#P=ProblemFixer(df)
#[P.remove_special_characters(k) for k,v in P.df.dtypes.items() if v=="object"]
#print(P.df.head())
#print(P.df.dtypes)
#P.df["timeDelivered"]=pd.to_datetime(P.df["timeDelivered"], errors='ignore',unit='s')
#print(P.df["timeDelivered"].head())

#P.remove_special_characters("string")
#print(P.df.head())
#df["string"]=df["string"].apply(lambda x: strip_accents(x)  )
#print(df.head())
# accented_string is of type 'unicode'


#P=ProblemFixer(df)
#print(P.df.head())

#log=pd.DataFrame(columns=['time', 'level', 'message'])
#log=log.append({'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'level': "INFO", 'message': "TEST"}, ignore_index=True)
#print(log)
#df = pd.DataFrame()
#df = df.append({'name': 'Zed', 'age': 9, 'height': 2}, ignore_index=True)
#print(df)
#P.extract_year_month_date("date")
#print(P.logger)
#P.logger_save("test_log.log")

#df['date'] = pd.to_datetime(df['date'], errors='ignore', unit='s')
#df["date"]=pd.to_datetime(df["date"], errors='ignore', dayfirst=False, utc=None,
#                                 box=True, format=None, coerce=False, unit='ns')




'''
A=ProblemFinder(df)
id_columns=A.columns_ids()
constant_columns=A.columns_constant_vales()
currency_columns=A.columns_currency()
msv_table_all=A.missing_zero_values_table()
mcv_columns=list(msv_table_all.index)


#* Check numeric columns
numeric_columns=A.get_numeric_columns()
if numeric_columns:
    #msv_table=msv_table_all.loc[numeric_columns,"Missing Values"].fillna(0)

    msv_dict={num_column:{"Missing values":int(msv_table_all.loc[num_column,"Missing Values"])  
                          if num_column in mcv_columns else 0} 
              for num_column in numeric_columns}
    msv_percent_dict={num_column:{"% Total Zero Missing Values":int(msv_table_all.loc[num_column,"% Total Zero Missing Values"])  
                          if num_column in mcv_columns else 0} 
              for num_column in numeric_columns}
        
    negv_dict={num_column:{"Negative values":int(calculate_negative_values(A.df[num_column]))}  
            for num_column in numeric_columns}
    outliers_dict={num_column:{"Outliers":int(calculate_numerical_outliers_Z(A.df,num_column))} 
                for num_column in numeric_columns}
    id_dict={ num_column:{"ID column":True if num_column in id_columns else False} 
            for num_column in numeric_columns  }
    const_dict={ num_column:{"Constant column":True if num_column in constant_columns else False} 
                for num_column in numeric_columns  }
    duplicates_dict={num_column:{"Number of duplicates":int(A.number_duplicates_column(num_column))} 
                    for num_column in numeric_columns}
    currency_dict={ num_column:{"Currency column":True if num_column in currency_columns else False} 
                for num_column in numeric_columns  }
    type_dict={num_column:{"Type":"Numeric"} for num_column in numeric_columns} 
    final_numeric={key: {**value,**negv_dict[key], **msv_percent_dict[key],
                    **outliers_dict[key],**id_dict[key], 
                    **const_dict[key],**duplicates_dict[key],
                    **currency_dict[key]
                    } for key, value in msv_dict.items()}
    #pprint.pprint(final_numeric)

#* check "object" columns
object_columns=A.type_columns_dict().get("object",None)
if object_columns:
    #msv_table=msv_table_all.loc[object_columns,"Missing Values"].fillna(0)
    msv_dict={num_column:{"Missing values":int(msv_table_all.loc[num_column,"Missing Values"])
                          if num_column in mcv_columns else 0} 
              for num_column in object_columns}
    id_dict={ num_column:{"ID column":True if num_column in id_columns else False} 
         for num_column in object_columns  }
    const_dict={ num_column:{"Constant column":True if num_column in constant_columns else False} 
            for num_column in object_columns  }
    duplicates_dict={num_column:{"Number of duplicates":int(A.number_duplicates_column(num_column))} 
                 for num_column in object_columns}
    currency_dict={ num_column:{"Currency column":True if num_column in currency_columns else False} 
            for num_column in object_columns  }
    type_dict={num_column:{"Type":"String"} for num_column in object_columns} 
    final_object ={key: {**value, **id_dict[key], 
                    **const_dict[key],**duplicates_dict[key],
                    **currency_dict[key],**type_dict[key]
                    } for key, value in msv_dict.items()}
    #pprint.pprint(final_object)

#* check datetype columns
datetime_columns=A.type_columns_dict().get("datetime64[ns]",None)

if datetime_columns:
    msv_dict={dt_column:{"Missing values":int(msv_table_all[dt_column]) if dt_column in mcv_columns else 0 } 
              for dt_column in datetime_columns}
    id_dict={ dt_column:{"ID column":True if dt_column in id_columns else False} 
         for dt_column in datetime_columns  }
    const_dict={ dt_column:{"Constant column":True if dt_column in constant_columns else False} 
            for dt_column in datetime_columns  } 
    #outliers_dict= A.datetime_ouliers(A.df[datetime_columns],2)
    type_dict={dt_column:{"Type":"Datetime"} for dt_column in datetime_columns} 
    final_datetime ={key: {**value, **id_dict[key], 
                    **const_dict[key],
    #                **outliers_dict[key],
                    **type_dict[key]
                    } for key, value in msv_dict.items()}


print(A.df.shape[1])
print(len(final_numeric.keys())+len(final_object.keys())+len(final_datetime.keys()))
all_col=A.df.columns
processed_col=list(final_numeric.keys())+list(final_object.keys())+list(final_datetime.keys())
print(list(set(all_col) - set(processed_col)))
final_all={**final_numeric,**final_object,**final_datetime}
final=dict()
final["file_name"]="retail_DS.csv"
final["Row duplicates"]=A.duplicates_in_dataset()
final["Columns"]=final_all
with open('problem_report_retail_DS.json', 'w') as outfile:
    json.dump(final, outfile)
'''







'''
df = pd.DataFrame({'A': ['7','7','9A'],
#                   'B': np.random.rand(3),
                   'C': ['foo','foo','baz'],
                   'D': ['who','who','when'],
                   "Currency":["123","123","€34234"]
                   })

_, idx = np.unique(df, axis = 1, return_index=True)
df = df.iloc[:, idx]
print(df)


col_list=[]
for col in df.columns:
    if df[col].dtype == "object":
        if (True in set(df[col].str.contains(r'(\£|\$|\€)' , regex=True).unique())):
            col_list.append(col)

col_currency=list(filter(lambda x: x != None,[ col 
                                       if True in set(df[col].str.contains(r'(\£|\$|\€)' , regex=True).unique()) 
                                       else None  
                                       for   col in df.columns if df[col].dtype == "object"] ))
print(col_currency)



#A.df = A.df.apply(lambda x: np.nan if isinstance(x, str) and (x.isspace() or not x) else x)
#A.df=A.df.replace(r'^\s*$', np.nan, regex=True)
#print(A.missing_zero_values_table().head().loc["Customer Email", : ])
#print(A.df["Customer Email"].unique())

#df.columns=[col.strip() for col in df.columns]
#print(df["Customer Email"].dtype)
#df=df.replace('', np.nan).apply(lambda x: x.astype(str).str.upper().str.strip() if x.dtype == "object" else x)
#print(df["Customer Email"].dtype)
#print(df["Customer Email"].unique())

#df = pd.DataFrame({'A': [7,-8,-9],
#                   'B': np.random.rand(3),
#                   'C': ['foo','bar','baz'],
#                   'D': ['1who','what','when']})

#print(calculate_negative_values(df["A"]))
#print(df)
#
#for col in df.columns:
#    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\D', '') , errors='coerce')
#print(df)
#print( 100 * df.isnull().sum() / len(df))
#print(df.dtypes)
#print(df.iloc[errors,:])

'''











'''
df = (pd.read_csv(filepath_or_buffer=os.path.join(data_path, 'master.csv'))
      .rename(columns={'suicides/100k pop' : 'suicides_per_100k',
                       ' gdp_for_year ($) ' : 'gdp_year', 
                       'gdp_per_capita ($)' : 'gdp_capita',
                       'country-year' : 'country_year'})
      .assign(gdp_year=lambda _df: _df['gdp_year'].str.replace(',','').astype(np.int64))
     )
'''