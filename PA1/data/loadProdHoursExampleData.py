import pandas as pd
                    
# read data for productivity /hours worked example
# note: prodhours2020.dat is a text containing the data in JMulTi format
# prodhours2020.dat is provided in the zip file on ILIAS
df = pd.read_csv('prodhours2020.dat', sep='\t', skiprows=1)
df.index = pd.date_range('1950', freq='Q', periods=len(df))# add date info

# print first and last few observations
print(df.head())
print(df.tail())
