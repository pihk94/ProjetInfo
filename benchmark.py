from Portefeuille import Portfeuille
from PM_strategy import CRP,BHP
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
CRP_cum_return = np.cumprod(y) - 1
y2= BHP(Port.df_normalized,START,END)
HBP_cum_return = np.cumprod(y2) - 1

x = pd.date_range(start=START,end=END,freq='30min')
plt.figure(figsize=(12,6))
sns.lineplot(x=x,y=CRP_cum_return,color = 'red')
sns.lineplot(x=x,y=HBP_cum_return,color = 'blue')
plt.legend(('UCRP','UHBP'),loc='upper right')
plt.title('Rendements cumulés pour différentes stratégies de gestion de portefeuille')
plt.show()