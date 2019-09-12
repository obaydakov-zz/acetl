

import pandas as pd
import numpy as np
import os


# to download https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
data_path = 'path/to/folder/'

df = (pd.read_csv(filepath_or_buffer=os.path.join(data_path, 'master.csv'))
      .rename(columns={'suicides/100k pop' : 'suicides_per_100k',
                       ' gdp_for_year ($) ' : 'gdp_year', 
                       'gdp_per_capita ($)' : 'gdp_capita',
                       'country-year' : 'country_year'})
      .assign(gdp_year=lambda _df: _df['gdp_year'].str.replace(',','').astype(np.int64))
     )

########## CONVERT COLUMN TO DATETIME
data['date'] = pd.to_datetime(data['date'])

#* CONVERT COLUMNS TO CATEGORY TYPE ###########
def convert_df(df: pd.DataFrame, deep_copy: bool = True):
    """Automatically converts columns that are worth stored as
    ``categorical`` dtype.
    Parameters
    ----------
    df: pd.DataFrame
        Data frame to convert.
    deep_copy: bool
        Whether or not to perform a deep copy of the original data frame.
    Returns
    -------
    pd.DataFrame
        Optimized copy of the input data frame.
    """
    return df.copy(deep=deep_copy).astype({
        col: 'category' for col in df.columns
        if df[col].nunique() / df[col].shape[0] < 0.5})
    


#* CHAINING DATAFRAME COMMANDS ###############
df = (pd.DataFrame({'a_column': [1, -999, -999],
                    'powerless_column': [2, 3, 4],
                    'int_column': [1, 1, -1]})
        .assign(a_column=lambda _df: _df['a_column'].replace(-999, np.nan),
                power_column=lambda _df: _df['powerless_column'] ** 2,
                real_column=lambda _df: _df['int_column'].astype(np.float64))
        .apply(lambda _df: _df.replace(4, np.nan))
        .dropna(how='all')
      )

#* PIPE OPERATION ####################
def log_head(df, head_count=10):
    print(df.head(head_count))
    return df

def log_columns(df):
    print(df.columns)
    return df

def log_shape(df):
    print(f'shape = {df.shape}')
    return df

(df
 .assign(valid_cy=lambda _serie: _serie.apply(
     lambda _row: re.split(r'(?=\d{4})', _row['country_year'])[1] == str(_row['year']),
     axis=1))
 .query('valid_cy == False')
 .pipe(log_shape)
)

#* DF NORMALIZATION ##############
from sklearn.preprocessing import MinMaxScaler
def norm_df(df, columns):
    return df.assign(**{col: MinMaxScaler().fit_transform(df[[col]].values.astype(float)) 
                        for col in columns})
for sex in ['male', 'female']:
    print(sex)
    print(
        df
        .query(f'sex == "{sex}"')
        .groupby(['country'])
        .agg({'suicides_per_100k': 'sum', 'gdp_year': 'mean'})
        .rename(columns={'suicides_per_100k':'suicides_per_100k_sum', 
                         'gdp_year': 'gdp_year_mean'})
#         Recommended in v0.25
#         .agg(suicides_per_100k=('suicides_per_100k_sum', 'sum'), 
#              gdp_year=('gdp_year_mean', 'mean'))
        .pipe(norm_df, columns=['suicides_per_100k_sum', 'gdp_year_mean'])
        .corr(method='spearman')
    )
    print('\n')

#* Fill NaN with the mean of the column
df['col'] = df['col'].fillna(df['col'].mean())
#* Drop any rows which have any nans
df.dropna()
#* Drop columns that have any nans
df.dropna(axis=1)
#* Only drop columns which have at least 90% non-NaNs
df.dropna(thresh=int(df.shape[0] * .9), axis=1)


#* np.where(if_this_is_true, do_this, else_do_that)
df['new_column'] = np.where(df['col'].str.startswith('foo') and  
                            not df['col'].str.endswith('bar'), 
                            True, 
                            df['col']) 


#* DEDUPTE STRING VALUES
# List of duplicate character names
contains_dupes = [
'Harry Potter', 
'H. Potter', 
'Harry James Potter', 
'James Potter', 
'Ronald Bilius \'Ron\' Weasley', 
'Ron Weasley', 
'Ronald Weasley']
# Print the duplicate values
process.dedupe(contains_dupes)
# Print the duplicate values with a higher threshold
process.dedupe(contains_dupes, threshold=90)

#* FUZZY MATCHING WITH DATETIME
from dateutil.parser import parse
dt = parse("Today is January 1, 2047 at 8:21:00AM", fuzzy=True)
print(dt)
2047-01-01 08:21:00


#* Making a list of missing value types
missing_values = ["n/a", "na", "--"]
df = pd.read_csv("property data.csv", na_values = missing_values)

#* Detecting numbers 
cnt=0
for row in df['OWN_OCCUPIED']:
    try:
        int(row)
        df.loc[cnt, 'OWN_OCCUPIED']=np.nan
    except ValueError:
        pass
    cnt+=1

#https://stackoverflow.com/questions/13682044/remove-unwanted-parts-from-strings-in-a-column
def try_extract(pattern, string):
    try:
        m = pattern.search(string)
        return int(m.group(0))
    except (TypeError, ValueError, AttributeError):
        return int(string)

p = re.compile(r'\d+')
df['A'] = [try_extract(p, x) for x in df['A']]

#* remove letters from numbers
df = pd.DataFrame({'A': [7,'AA8qwe','9A'],
                   'B': np.random.rand(3),
                   'C': ['foo','bar','baz'],
                   'D': ['who','what','when']})
#print(df)
#
df['A'] = df['A'].astype(str).str.replace(r'\D', '').astype(int)
print(df)
print(df.dtypes)

#* COLORING OUTPUT
from clint.textui import colored
print(colored.red('some warning message'))
print(colored.green('nicely done!'))

from clint.textui import prompt, validators
path = prompt.query('Installation Path', default='/usr/local/bin/', validators=[validators.PathValidator()])