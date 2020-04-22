import numpy as np
import tensorflow as tf
from Agent import Agent
from Portefeuille import Portfeuille
from Buffer import Buffer
class Portfolio_managment:
    def __init__(self, symbols,period_start,period_end,mode):
        self.symbols = symbols
        self.period_start = period_start
        self.period_end = period_end
        self.symbols_num = len(symbols)
        #HYPER PARAMETERS
        self.BUFFER_SIZE = 200
        self.BATCH_SIZE = 10
        self.SHOW_EVERY = 250
        self.WINDOW_SIZE = 50
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
        for episode in range(episode_depart+1,episode_depart+episode_fin+1):
            Port = Portfeuille(self.symbols,self.period_start,self.period_end)
            #Préparation des états
            state = 
            cum_return = 1
            #Deuxieme boucle sur toute la période
            for step in range(len(state)-2):
                if step == 0:
                    last_action = np.ones(len(self.state_dim[0]))
                else:
                    last_action = np.array(Port.weights[-1][:self.state_dim[0]])
                action = self.agent.model.predict([state[step].reshape([1,self.state_dim[2],self.state_dim[1],self.state_dim[0]]),
                last_action.reshape([1,self.state_dim[0]]),np.array([[self.CASH_BIAS]])])
                rendement_jour, futur_price = Port.get_return(action[0],last_action,step)
                self.replay(self.agent,self.buffer,self.BATCH_SIZE,state,step,futur_price,last_action)
                cum_return *= rendement_jour
                self.total_step +=1
                if step % self.SHOW_EVERY:
                    print(f"Episode {ep}, pas {step}\nCumReturn {cum_return} à la date : {state[step]}")
                    print(action[0])
            self.buffer.clear()
            self.episode_reward.append(cum_return)
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