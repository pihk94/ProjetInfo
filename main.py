import datetime
from simulation import Portfolio_managment
SYMBOLS = ['ETHBTC','XRPBTC','EOSBTC','LTCBTC','ZECBTC','ETCBTC','XMRBTC']
START = datetime.datetime(2019,1,1)
END = datetime.datetime(2019,12,31)
PM = Portfolio_managment(SYMBOLS,START,END,'train')
PM.simulate(episode_depart=0,episode_fin=5)