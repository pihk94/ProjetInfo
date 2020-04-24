import datetime
from simulation import Portfolio_managment

#Module pour lancer une simulation
SYMBOLS = ['ETHBTC','XRPBTC','EOSBTC','LTCBTC','ZECBTC','ETCBTC','XMRBTC']
START = datetime.datetime(2017,8,1)
END = datetime.datetime(2019,12,31)
PM = Portfolio_managment(SYMBOLS,START,END,'train',LR=0.0002)
PM.simulate(episode_depart=0,episode_fin=100)