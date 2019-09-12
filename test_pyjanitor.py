import pandas as pd
import janitor
import pandas_flavor as pf
import numpy as np


company_sales = {
    'SalesMonth': ['Jan', 'Feb', 'Mar', 'April'],
    'Company1': [150.0, 200.0, 300.0, 400.0],
    'Company2': [180.0, 250.0, np.nan, 500.0],
    'Company3': [400.0, 500.0, 600.0, 675.0]
}

df = (
    pd.DataFrame.from_dict(company_sales)
    .remove_columns(['Company1'])
    .dropna(subset=['Company2', 'Company3'])
    .rename_column('Company2', 'Amazon')
    .rename_column('Company3', 'Facebook')
    .add_column('Google', [450.0, 550.0, 800.0])
)

print(df.head())

https://pyjanitor.readthedocs.io/notebooks/pyjanitor_intro.html#The-pyjanitor-approach

aws lambda publish-layer-version --layer-name pyjanitor \
     --description "janitor for data cleaning" \
     --zip-file pyjanitor.zip \
     --compatible-runtimes python3.6
