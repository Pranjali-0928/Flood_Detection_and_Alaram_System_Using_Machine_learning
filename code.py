import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#--------load data set-------------------------
filename = "kerala.csv"
df = pd.read_csv(filename)



#-------------------show data------------------
print(df.info())
print("Annual rainfall",df[' ANNUAL RAINFALL'])




#--------------graph of data---------------------
df.set_axis(df['YEAR'], inplace=True)
plt.plot(df['YEAR'],df[' ANNUAL RAINFALL'])
plt.xlabel('YEAR')
plt.ylabel('ANNUAL RAINFALL')
plt.show()

