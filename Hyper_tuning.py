import datetime
from simulation import Portfolio_managment
import pandas as pd
#Le but de ce fichier est d'optimiser les différents paramètres possible et de choisir un réseau optimum
#Nous voulons tester les effets de plusieurs paramètres, par soucis de capacité de calcul, dommage que le cours de données distribués aient commencé tardivement
# Effets de variation sur les paramètres suivant : 
# Learning Rate
# Performance des différents models
# WINDOW_SIZE

#TIME RANGE 
# START = datetime.datetime(2017,8,1,0,0) #Premiere fois ou on a des données pour tous
# END = datetime.datetime(2018,7,31,0,0) #On prend un an d'expérience
# SYMBOLS = ['ETHBTC','XRPBTC','EOSBTC','LTCBTC','ZECBTC','ETCBTC','XMRBTC'] #TOP DES CURRENCIES QUE L'ON A PU RECUPERER AISEMENT
# MODELS = ['CNN','CNN_LSTM'] #Les trois modèles disponibles
# WINDOW_SIZE = [48,96,48*7]
# LR = [2e-5,9e-5,2e-4]
# #ON POURRAIT PRENDRE DES PLAGES PLUS GRANDES, FAIRE D AUTRES PARAMETRES MAIS CA PRENDS DU TEMPS A COMPUTE, en fait le LSTM est trop long
# lst = []
# for model in MODELS:
#     for window in WINDOW_SIZE:
#         j=0
#         for lr in LR:
#             j+=1
#             PF = Portfolio_managment(SYMBOLS,START,END,'train',WINDOW_SIZE=window,LR=lr,MODEL_NAME=model)
#             episode_reward,all_ = PF.simulate(episode_depart=0,episode_fin=20)
#             lst += [(model,window,lr,episode_reward)]
#             print('\nENDED\n')
#             exp1 = pd.DataFrame(lst,columns=['model','window','lr','epsiode_reward'])
#             exp1.to_csv('tuning.csv')
#             print('export 1')
#             exp2 =pd.DataFrame(all_,columns=['cum_rwd','SR','SR_BTC','MD','VOL'])
#             exp2.to_csv(f'tuning_{model}_{window}_{j}.csv')
#             print('export 2')

#Désormais que nous savons que nous allons choisir LR = 0.0002 et une window de 48 jours, tentons de tester différentes rewards functions
START = datetime.datetime(2017,8,1,0,0)
END = datetime.datetime(2018,7,31,0,0)
SYMBOLS = ['ETHBTC','XRPBTC','EOSBTC','LTCBTC','ZECBTC','ETCBTC','XMRBTC'] #TOP DES CURRENCIES QUE L'ON A PU RECUPERER AISEMENT
MODELS = ['CNN']
REWARDS = ['avg_uniform_constant_rebalanced','SR','calmar_ratio','avg_log_cum_return']
lst = []
j=0
for model in MODELS:
    for reward in REWARDS:
        j+=1
        PF = Portfolio_managment(SYMBOLS,START,END,'train',WINDOW_SIZE=48,LR=0.0002,MODEL_NAME=model,reward=reward)
        episode_reward,all_ = PF.simulate(episode_depart=0,episode_fin=20)
        lst += [(model,reward,episode_reward)]
        exp1 = pd.DataFrame(lst,columns=['model','reward','epsiode_reward'])
        exp1.to_csv(f'tuning_{reward}.csv')
        exp2 =pd.DataFrame(all_,columns=['cum_rwd','SR','SR_BTC','MD','VOL'])
        exp2.to_csv(f'tuning_{model}_{j}_{reward}.csv')
