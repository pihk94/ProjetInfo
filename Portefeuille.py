import pandas as pd
import requests
import time
import datetime
from datetime import timedelta
import numpy as np
import os


class Portfeuille:
    def __init__(self,symbols,start,end,cash):
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
        self.returns = [1]
        self.transition_factor = 0.002
        #Chargement des données et on fait en sorte qu'on soit sure qu'on ast le même format pour tous
        for symbol in self.symbols + [cash]:
            if not os.path.exists("Data/"+symbol+".csv"):
                print('Collecte des donnéees pour ',symbol)
                self.extract_hist_curr(symbol,"30m",10000,datetime.datetime(2014,1,1),datetime.datetime.now(),-1,False)
            else:
                #print(f"Présence des données historiques pour {symbol}")
                self.make_format("Data/"+symbol+".csv")
        self.cash = pd.read_csv('Data/'+cash+'.csv')
        self.cash.time = pd.to_datetime(self.cash.time)
        self.cash = self.cash[(self.cash.time <= end) & (self.cash.time >= start+timedelta(minutes=60*24*7)+timedelta(minutes=30))]
        self.cash = (self.cash.open.shift(-1)/self.cash.open).fillna(1)
        #On collecte chaque features dans un seul dataframe
        self.df_close = self.extract_column(self.symbols,self.start,self.end,label ="close")
        self.df_high = self.extract_column(self.symbols,self.start,self.end,label ="high")
        self.df_low = self.extract_column(self.symbols,self.start,self.end,label ="low")
        self.df_open = self.extract_column(self.symbols,self.start,self.end,label ="open")
        self.df_volume = self.extract_column(self.symbols,self.start,self.end,label ="volume")
        self.df_normalized =(self.df_open.shift(-1) / self.df_open).fillna(1)
        self.idx_depart = np.where(self.df_close.index == start+timedelta(minutes=60*24*7))[0][0]
    def extract_column(self,symbols,start,end,label):
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
            df = df[(df.time <= end) & (df.time >= start)]
            idx = df.time.values
            df = pd.DataFrame(df[label].values,columns=[symbol])
            df.set_index(idx,inplace=True)
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
        start = time.mktime(start.timetuple())*1000
        end = time.mktime(end.timetuple())*1000
        data = []
        step = 1000*60*limit
        start = start - step
        while start < end:
            start +=step
            fin = start + step
            r = requests.get(f'https://api.bitfinex.com/v2/candles/trade:{interval}:t{symbol.upper()}/hist?limit={limit}&start={start}&end={end}&sort={sort}').json()
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
        print('Travail terminé, fichier enregistré : {}-{}.csv'.format(os.getcwd(),symbol))
        h_fin = datetime.datetime.now()
        print('Début : {} | Fin : {}.\nExécution {} minutes.'.format(h_debut,h_fin,-1*(time.mktime(h_debut.timetuple())-time.mktime(h_fin.timetuple()))/60))
    def make_format(self,filename):
        """
            Permet l'obtention du bon format pour le fichier.
        """
        df= pd.read_csv(filename)
        df.time = pd.to_datetime(df.time)
        dt_range = pd.date_range(start=datetime.datetime(2017,8,1,0,0),end=datetime.datetime(2020,4,10,0,0),freq='30min')
        dt_range = pd.DataFrame(dt_range,columns=["time"])
        df = pd.merge(dt_range,df,how="left",on="time")
        df = df.fillna(method="ffill")
        df[['time','open','close','high','low','volume']].to_csv(filename,index=False)
    def clear(self):
        """
            RAZ des valeurs 
        """
        self.weights = []
        self.returns = [1]
    def get_return(self, weight, last_weight, step):
        """
            Calcul des returns
        """
        cout_transaction = 1 - self.transition_factor * np.sum(np.abs(weight[:-1] - last_weight))
        futur_price = np.append(self.df_normalized.iloc[self.idx_depart + step].values, [1])
        return_journalier = np.dot(futur_price,weight) * cout_transaction
        self.returns.append(return_journalier)
        self.weights.append(weight)
        return return_journalier,futur_price
