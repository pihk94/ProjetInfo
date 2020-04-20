import pandas as pd
import requests
import time
import datetime
import numpy as np
import os


class Portfeuille:
    def __init__(self,symbols,start,end):
        """
            Input: 
                symbols :  Indices (pairs) des cryptos
                    type : liste
                end : Date de fin
                    type : datetime
                start : Date de début
                    type : datetime
                label : Nom de la colonne doit être parmis la liste [close,high,low,open,volume]
                    type : string
        """
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.start = start
        self.end = end
        self.weights = []
        self.returns = []
        self.transition_factor = 0.002
        #Chargement des données et on fait en sorte qu'on soit sure qu'on ast le même format pour tous
        for symbol in self.symbols:
            if not os.path.exists("Data/"+symbol+".csv"):
                self.extract_hist_curr(symbol,"30m",10000,datetime.datetime(2014,1,1),datetime.datetime.now(),-1,False)
            else:
                print(f"Présence des données historiques pour {symbol}")
                self.make_format("Data/"+symbol+".csv")
        #On collecte chaque features dans un seul dataframe
        self.df_close = self.extract_column(self.symbols,self.start,self.end,label ="close")
        self.df_high = self.extract_column(self.symbols,self.start,self.end,label ="high")
        self.df_low = self.extract_column(self.symbols,self.start,self.end,label ="low")
        self.df_open = self.extract_column(self.symbols,self.start,self.end,label ="open")
        self.df_volume = self.extract_column(self.symbols,self.start,self.end,label ="volume")
    def extract_column(self,symbols,end,start,label):
        """
            Input: 
                symbols :  Indices (pairs) des cryptos
                    type : liste
                end : Date de fin
                    type : datetime
                start : Date de début
                    type : datetime
                label : Nom de la colonne doit être parmis la liste [close,high,low,open,volume]
                    type : string
            Output:
                Dataframe contenant toutes les cryptos du portfolio entre deux dates selon une colonne
        """
        full_df = pd.DataFrame()
        for symbol in symbols:
            df = pd.read_csv("Data/"+symbol+".csv")
            df.time = pd.to_datetime(df.time)
            df[(df.time <= end) & (df.time >= start)]
            df = pd.DataFrame(df[label].values,columns=[symbol])
            if full_df.empty:
                full_df = df
            else:
                full_df = full_df.join(df, how='outer')
        return full_df
    def extract_hist_curr(self,symbol,interval,limit,start,end,sort,verbose=True):
        """ Description :
        INPUTS :
            symbole : Pair currency ( disponible ici : https://coinmarketcap.com/exchanges/bitfinex/)
            interval :  '1m', '5m', '15m', '30m', '1h', '3h', '6h', '12h', '1D', '7D', '14D', '1M'
            limit : max 10000
            start : Date de début au format datetime
            end : Date de fin au format datetime
            sort : Ordre chronologique descendant si 1, si -1 ascendant
        OUTPUT :
            data : DataFrame avec l'historique des valeurs
        """
        h_debut = datetime.datetime.now()
        debut = start
        start = time.mktime(start.timetuple())*1000
        ended = end
        end = time.mktime(end.timetuple())*1000
        data = []
        step = 1000*60*limit
        start = start - step
        while start < end:
            start +=step
            fin = start + step
            r = requests.get('https://api.bitfinex.com/v2/candles/trade:{}:t{}/hist?limit={}&start={}&end={}&sort=-1'.
                                    format(interval, symbol.upper(), limit, start, fin, sort)).json()
            data.extend(r)
            if verbose == True:
                print('Extraction des données de la période {} à {} pour {}. Taille de la requete {}'.format(pd.to_datetime(start,unit='ms'),pd.to_datetime(fin,unit='ms'),symbol,len(r)))
            time.sleep(1.5)
        ind = [np.ndim(x) != 0 for x in data]
        data = [i for (i, v) in zip(data, ind) if v]
        names = ['time', 'open', 'close', 'high', 'low', 'volume']
        df = pd.DataFrame(data, columns=names)
        df.drop_duplicates(inplace=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        df.index = pd.to_datetime(df.index, unit='ms')
        df.to_csv('Data/{}.csv'.format(symbol))
        print('Travail terminé, fichier enregistré : {}\{}.csv'.format(os.getcwd(),symbol))
        h_fin = datetime.datetime.now()
        print('Début : {} | Fin : {}.\nExécution {} minutes.'.format(h_debut,h_fin,-1*(time.mktime(h_debut.timetuple())-time.mktime(h_fin.timetuple()))/60))
    def make_format(self,filename):
        df= pd.read_csv(filename)
        df.time = pd.to_datetime(df.time)
        dt_range = pd.date_range(start=datetime.datetime(2017,8,1,0,0),end=datetime.datetime(2020,4,10,0,0),freq='30min')
        dt_range = pd.DataFrame(dt_range,columns=["time"])
        df = pd.merge(dt_range,df,how="left",on="time")
        df = df.fillna(method="ffill")
        df[['time','open','close','high','low','volume']].to_csv(filename,index=False)
    def clear(self):
        self.weights = []
        self.returns = []
    def get_return(self, weights, last_weights, step):

#SYMBOLS = ['BTCUSD','ETHUSD','XRPUSD','EOSUSD','LTCUSD','BCHUSD','ZECUSD','ETCUSD','NEOUSD','XMRUSD']
SYMBOLS = ['BTCUSD','ETHBTC','XRPBTC','EOSBTC','LTCBTC','ZECBTC','ETCBTC','XMRBTC']
Portfeuille(SYMBOLS,None,None)
