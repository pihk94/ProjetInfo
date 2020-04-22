import numpy as np
import pandas as pd
import tensorflow as tf
from Agent import Agent
from Portefeuille import Portfeuille
from Buffer import Buffer
from tqdm import tqdm
import pickle
import os 
class Portfolio_managment:
    def __init__(self, symbols,period_start,period_end,mode):
        tf.compat.v1.disable_eager_execution()
        self.symbols = symbols
        self.period_start = period_start
        self.period_end = period_end
        self.symbols_num = len(symbols)
        self.mode = mode
        #HYPER PARAMETERS
        self.BUFFER_SIZE = 200
        self.BATCH_SIZE = 10
        self.SHOW_EVERY = 24*7*2*4
        self.WINDOW_SIZE = 48 # Une journée
        self.CASH_BIAS = 0 # ???
        self.NB_FEATURES = 3
        self.SAMPLE_BIAS = 1.05 # ???
        self.state_dim = (self.symbols_num,self.WINDOW_SIZE,self.NB_FEATURES)
        self.action_size = self.symbols_num +1
        self.LR_list = {'train':2e-5,'test':9e-5,'valid':9e-5}
        self.ROLLING_STEPS_dic = {'train':1,'test':0,'valid':0}
        self.ROLLING_STEPS = self.ROLLING_STEPS_dic[mode]
        self.LR = self.LR_list[mode]
        #Initialisation 
        self.episode_reward = []
        self.total_step = 0
        self.session = self.tf_session()
        np.random.seed(4)
        self.agent = Agent(self.session,self.state_dim,self.action_size,self.BATCH_SIZE,self.LR,'avg_log_cum_return','CNN')
        self.buffer = Buffer(self.BUFFER_SIZE, self.SAMPLE_BIAS)
    def tf_session(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        tf.compat.v1.keras.backend.set_learning_phase(1)
        return sess
    def simulate(self,episode_depart,episode_fin):
        #Premiere boucle sur les épisodes
        cum_return_all = []
        for episode in tqdm(range(episode_depart+1,episode_depart+episode_fin+1)):
            if not (self.mode == 'train' and episode !=episode_depart):
                self.charger_poids(episode)
            Port = Portfeuille(self.symbols,self.period_start,self.period_end)
            #Préparation des états
            state = self.states(Port,self.WINDOW_SIZE)
            cum_return = 1
            #Deuxieme boucle sur toute la période
            for step in range(len(state)-2):
                if step == 0:
                    last_action = np.ones(self.state_dim[0])
                else:
                    last_action = np.array(Port.weights[-1][:self.state_dim[0]])
                reshape1 = state[step].reshape([1,self.state_dim[2],self.state_dim[1],self.state_dim[0]])
                reshape2 = last_action.reshape([1,self.state_dim[0]])
                reshape3 =np.array([[self.CASH_BIAS]])
                action = self.agent.model.predict([reshape1,reshape2,reshape3])
                rendement_jour, futur_price = Port.get_return(action[0],last_action,step)
                self.replay(state,step,futur_price,last_action)
                cum_return *= rendement_jour
                self.total_step +=1
                if not step % self.SHOW_EVERY:
                    print(f"Episode {episode}, pas {step}\nCumReturn {cum_return} à la date : {Port.df_close.iloc[Port.idx_depart + step+1].name}")
                    print(action[0])
            self.buffer.clear()
            self.episode_reward.append(cum_return)
            self.enregistrer_poids(episode,Port,cum_return)
            cum_return_all.append(cum_return)
        pd.DataFrame(cum_return_all,columns=['cum_return']).to_csv('Model/Resultat/cum_return.csv')
        return self.episode_reward
    def replay(self,state,step,futur_price,last_action):
        self.buffer.add(state[step],futur_price,last_action)
        for _ in range(self.ROLLING_STEPS):
            batch,current_batch_size = self.buffer.getBatch(self.BATCH_SIZE)
            states = np.asarray([i[0] for i in batch])
            futur_prices = np.asarray([i[1] for i in batch])
            last_actions = np.asarray([i[2] for i in batch])
            cash_biass = np.array([[self.CASH_BIAS] for _ in range(current_batch_size)])
            if step > 10:
                self.agent.train(states,last_actions,futur_prices,cash_biass)
    def states(self,portefeuille,window_size):
        state = []
        df = np.array([portefeuille.df_close.values,portefeuille.df_high.values,portefeuille.df_low.values], dtype='float')
        for j in range(portefeuille.idx_depart -1, len(df[0])):
            temp = np.copy(df[:, j-window_size+1:j+1 , :])
            for feature in range(df.shape[0]):
                for k in range(portefeuille.num_symbols):
                    if temp[feature,-1,k] == 0:
                        temp[feature,:,k] /= temp[feature,-2,k]
                    else:
                        temp[feature,:,k] /= temp[feature,-1,k]
            state.append(temp)
        return state
    def charger_poids(self,ep):
        if self.mode == 'train' or self.mode =="valid":
            self.agent.model.load_weights(f"Model/Agent_poids_{self.mode}_ep_{ep-1}.h5")
        else:
            self.agent.model.load_weights(f"Model/Agent_poids_test_ep_{ep-1}.h5")
    def enregistrer_poids(self,ep,Port,cum_return):
        if self.mode == "train":
            self.agent.model.save_weights(f"Model/Agent_poids_{self.mode}_ep_{ep}.h5", overwrite=True)
        elif self.mode =="valid":
            self.agent.model.save_weights(f"Model/Agent_poids_{self.mode}_ep_{ep-1}.h5", overwrite=True)
        filename = 'Portfolio_{}_{}_{}_{}_{}.pickle'.format(self.mode,str(Port.start)[:10],str(Port.end)[:10],'-'.join(Port.symbols), cum_return)
        with open("Model/Resultat/{}".format(filename),"wb") as outfile:
            pickle.dump(Port, outfile)
            print("Sauvegardé sous {}".format(filename) )
