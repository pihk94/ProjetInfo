from Portefeuille import Portfeuille
from PM_strategy import CRP
import seaborn as sns
import numpy as np 
import pandas as pd
import datetime
import matplotlib.pyplot as plt

SYMBOLS = ['ETHBTC','XRPBTC','EOSBTC','LTCBTC','ZECBTC','ETCBTC','XMRBTC']
START = datetime.datetime(2017,8,1)
END = datetime.datetime(2020,4,1)
Port = Portfeuille(SYMBOLS,START,END)
y = CRP(Port.df_normalized,START,END)
x = pd.date_range(start=START,end=END,freq='30min')
sns.lineplot(x=x,y=y)
plt.show()
