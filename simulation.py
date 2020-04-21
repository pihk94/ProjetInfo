import numpy as np
import tensorflow as tf

class Portfolio_managment:
    def __init__(self, symbols,period_start,period_end):
        self.symbols = symbols
        self.period_start = period_start
        self.period_end = period_end
        self.symbols_num = len(symbols)
        #HYPER PARAMETERS
        self.BUFFER_SIZE = 200
        self.BATCH_SIZE = 10
        self.WINDOW_SIZE = 50
        self.CASH_BIAS = 0 # ???
        self.NB_FEATURES = 3
        self.DEPENDANT_FACTOR = 0 # ???
        self.state_dim = (self.symbols_num,self.WINDOW_SIZE,self.NB_FEATURES)
        #Initialisation 
        self.episode_reward = []
        self.total_step = 0
        self.session = self.tf_session()
        np.random.seed(4)
    def Qdl(self,start,end,episode_depart,episode_fin):
        pass
    def tf_session(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        tf.compat.v1.keras.backend.set_learning_phase(1)
        return sess