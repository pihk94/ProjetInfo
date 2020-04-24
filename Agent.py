import tensorflow as tf
from tensorflow.keras.layers import Input,Convolution2D,Reshape,concatenate,multiply,Flatten,Activation,Permute,Dense,TimeDistributed,LSTM
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
class Agent:
    """
        Dans cette objet est défini le réseau de neurone choisit et son fonctionnement avec ses différentes fonctions de reward possible.
        A noter qu'il était encore difficile pour nous d'implémenter des réseaux de neurones complexes car nous n'avons pas vraiment d'expérience dans ce domaine.
        De ce fait, deux réseaux sont implémentés, le CNN et le CNN + LSTM. 
        Le problème est que pour le dernier, le temps d'éxécution moyen est très très long... De ce fait, nous utilisons uniquement le CNN.
        
        Input:
            session :  Session permettant d'utiliser le GPU, tf v1
                type : tensorflow session method
            state_size : Tuple de trois dimensions contenant le (nombre de crypto,la taille de la fenetre,le nombre de variable)
                type : tuple
            action_size : Nombre d'actions réalisable (nombre de cryptos +1)
                type : integer
            BATCH_SIZE : Taille du batch
                type : integer
            LR : Learning Rate
                type :float
            reward : Function de reward à utiliser pour l'optimisation, doit être une valeur dans ['avg_uniform_constant_rebalanced','SR','calmar_ratio','avg_log_cum_return']
                type : string
            NN : Architecture du réseau de neurones à utiliser CNN ou CNN_LSTM (ce dernier est déconseillé vu le temps d'exécution)
                type : string
    """
    def __init__(self,session,state_size,action_size,BATCH_SIZE,LR,reward,NN):
        self.sess = session
        self.state_size = state_size
        self.action_size = action_size
        #Hyperparamètres modifiable
        self.BATCH_SIZE = BATCH_SIZE
        self.DECAY_STEPS = 50000
        self.DECAY_RATE = 0.1
        self.GLOBAL_STEP = tf.Variable(0, trainable=False,name='global_step')
        self.LR = tf.compat.v1.train.exponential_decay(LR,self.GLOBAL_STEP,self.DECAY_STEPS,self.DECAY_RATE, staircase=False) # diminue le LR au cours du training, tous les 
        self.TRANSITION_FACTOR = 0.002
        tf.compat.v1.keras.backend.set_session(self.sess)
        self.futur_price = tf.compat.v1.placeholder(tf.float32,[None,state_size[0]+1])
        #Initialisation du model
        if NN == "CNN":
            print('Construction du CNN')
            self.model,self.weights,self.state,self.last_action,self.cash_bias,self.test = self.CNN(state_size)
        elif NN == 'CNN_DENSE':
            print(f'Construction du {NN}')
            self.model,self.weights,self.state,self.last_action,self.cash_bias,self.test = self.CNN_DENSE(state_size)
        elif NN == 'CNN_LSTM':
            print(f'Construction du {NN}')
            self.model,self.weights,self.state,self.last_action,self.cash_bias,self.test = self.CNN_LSTM(state_size)
        else:
            raise ValueError("Le réseau n'est pas implémenté.")
        #Choix de la fonction de récompense
        if reward == 'avg_log_cum_return':
            self.reward = self.avg_log_cum_return()
        elif reward == 'avg_uniform_constant_rebalanced':
            self.reward = self.avg_uniform_constant_rebalanced()
        elif reward == 'SR':
            self.reward = self.SR()
        elif reward == 'calmar_ratio':
            self.reward = self.calmar_ratio()
        else:
            raise ValueError('Fonction de reward non implémenté.')
        #On prend l'optimiser ADAM qui est le plus souvent utilisé dans la littérature et le plus efficace
        self.optimizer = tf.compat.v1.train.AdamOptimizer(LR).minimize(self.reward,global_step =self.GLOBAL_STEP)
        self.sess.run(tf.compat.v1.global_variables_initializer())
    def avg_log_cum_return(self):
        print("Reward : Rendements cumulés moyen logarithmique")
        self.cout_transaction = 1 - tf.reduce_sum(self.TRANSITION_FACTOR * tf.abs(self.model.output[:,:-1] - self.last_action), axis=1)
        return -tf.reduce_mean(tf.math.log(self.cout_transaction * tf.reduce_sum(self.model.output * self.futur_price,axis=1)))
    def avg_uniform_constant_rebalanced(self):
        print('Reward : avg_uniform_constant_rebalanced')
        self.cout_transaction = 1 - tf.reduce_sum(self.TRANSITION_FACTOR * tf.abs(self.model.output[:,:-1] - self.last_action), axis=1)
        return -tf.reduce_mean(tf.math.log(self.cout_transaction * tf.reduce_sum(self.model.output * self.futur_price,axis=1)/tf.reduce_sum(self.futur_price[:,:-1] / self.state_size[0],axis = 1)))
    def SR(self):
        print('Reward : Ratio de Sharpe')
        self.cout_transaction = 1 - tf.reduce_sum(self.TRANSITION_FACTOR * tf.abs(self.model.output[:,:-1] - self.last_action), axis=1)
        self.avg_return = tf.math.log(self.cout_transaction * tf.reduce_sum(self.model.output * self.futur_price ,axis=1))
        # On considere notre free risque rate = 0 (grosse simplification)
        return -(tf.reduce_mean(self.avg_return) * tf.sqrt(tf.compat.v1.to_float(tf.size(self.avg_return)))) / tf.compat.v1.keras.backend.std(self.avg_return)
    def calmar_ratio(self):
        print('Reward : Ratio de Calmar')
        self.cout_transaction = 1 - tf.reduce_sum(self.TRANSITION_FACTOR * tf.abs(self.model.output[:,:-1] - self.last_action), axis=1)
        self.avg_return = tf.math.log(self.cout_transaction * tf.reduce_sum(self.model.output * self.futur_price ,axis=1))
        self.cum_return = tf.math.cumprod(self.avg_return)
        self.max_rolling = tf.scan(lambda a,x : tf.math.maximum(a,x), self.cum_return)
        self.drawdown = (self.max_rolling - self.cum_return) / self.max_rolling
        return -(tf.reduce_prod(self.avg_return)-1)/tf.reduce_max(self.drawdown)
    def CNN(self,state_size):
        """
            Réseau de neurones à trois couches : 3 convolutions
        """
        #HYPERPARAMETRES
        Kernel2D_1 = (5,1)
        Filter2D_1 = 3 # Valeur souvent utilisé pour commencer
        Kernel2D_2 = (state_size[1] - Kernel2D_1[0]+1,1)
        Filter2D_2 = 20
        State = Input(shape=[state_size[2],state_size[1],state_size[0]])
        last_action = Input(shape=[state_size[0]])
        last_action_1 = Reshape((1, 1, state_size[0]))(last_action)
        cash_bias = Input(shape=[1])
        #Construction des couches du réseau
        Conv2D_1 = Convolution2D(
            batch_input_shape=(self.BATCH_SIZE, state_size[2], state_size[1], state_size[0]),  
            filters=Filter2D_1,
            kernel_size=Kernel2D_1,     
            strides=1,
            padding='valid',              
            data_format='channels_first',     
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer=regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(State)
        #Deuxieme couche
        Conv2D_2 = Convolution2D(
            batch_input_shape= (self.BATCH_SIZE, Filter2D_1, state_size[1]-Kernel2D_1[0]+1, state_size[0]),
            filters=Filter2D_2,
            kernel_size=Kernel2D_2,     
            strides=1,
            padding='valid',                   
            data_format='channels_first',      
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer= regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(Conv2D_1)
        Concate = concatenate([Conv2D_2, last_action_1], axis=1)
        #Troisieme couche
        Conv2D_3 = Convolution2D(
            batch_input_shape=(self.BATCH_SIZE, Filter2D_2+1, 1, state_size[0]),
            filters=1,
            kernel_size=(1, 1),              
            strides=1,
            padding='valid',                 
            data_format='channels_first',     
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer=regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
        )(Concate)
        vote = Flatten()(Conv2D_3)#On rend l'output en une seule dimension
        F1 = concatenate([vote,cash_bias],axis=1)
        #Préparation de l'output,i.e nos actions. On utilise la fonction softmax comme activation
        action = Activation('softmax')(F1)
        model = Model(inputs=[State,last_action,cash_bias],outputs=action)
        return model, model.trainable_weights, State, last_action,cash_bias,[vote,F1]
    def CNN_DENSE(self, state_size):
        #Reprise de l'architecture du CNN + ajout d'une couche dense de 100
        #HYPERPARAMETRES
        Kernel2D_1 = (5,1)
        Filter2D_1 = 3 # Valeur souvent utilisé pour commencer
        Kernel2D_2 = (5,1)
        Filter2D_2 = 20
        DENSE = 100
        State = Input(shape=[state_size[2],state_size[1],state_size[0]])
        last_action = Input(shape=[state_size[0]])
        last_action_1 = Reshape((state_size[0], 1))(last_action)
        cash_bias = Input(shape=[1])
        #Ajout des couches
        Conv2D_1 = Convolution2D(
            batch_input_shape=(self.BATCH_SIZE, state_size[2], state_size[1], state_size[0]),  
            filters=Filter2D_1,
            kernel_size=Kernel2D_1,     
            strides=1,
            padding='valid',              
            data_format='channels_first',     
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer=regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(State)
        Conv2D_2 = Convolution2D(
            batch_input_shape= (self.BATCH_SIZE, Filter2D_1, state_size[1]-Kernel2D_1[0]+1, state_size[0]),
            filters=Filter2D_2,
            kernel_size=Kernel2D_2,     
            strides=1,
            padding='valid',                   
            data_format='channels_first',      
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer= regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(Conv2D_1)
        #Troisieme couche dense
        L1 = Permute((3,1,2))(Conv2D_2)
        L2 = Reshape((state_size[0], -1))(L1)
        L3 = concatenate([last_action_1, L2], axis=2)
        print(Conv2D_1,Conv2D_2)
        print(state_size,L1,L2,last_action_1,L3)
        L4 = TimeDistributed(Dense(DENSE, activation='relu'), input_shape=[state_size[0],-1])(L3)
        L5 = TimeDistributed(Dense(1, activation='linear'), input_shape=[state_size[0], DENSE])(L4)
        vote = Flatten()(L5)
        F1 = concatenate([vote,cash_bias],axis=1)
        #Fonction d'activation de l'ouput final softmax
        action = Activation('softmax')(F1)
        model = Model(inputs=[State, last_action, cash_bias], outputs=action)
        return model, model.trainable_weights, State, last_action, cash_bias, F1
    def CNN_LSTM(self, state_size):
        """
            Deux couches de convolutions puis deux de LSTM
        """
        #On change juste la dernière couche
        #HYPERPARAMETRES
        Kernel2D_1 = (5,1)
        Filter2D_1 = 3 # Valeur souvent utilisé pour commencer
        Kernel2D_2 = (5,1)
        Filter2D_2 = 20
        Lstm = 10
        State = Input(shape=[state_size[2],state_size[1],state_size[0]])
        last_action = Input(shape=[state_size[0]])
        cash_bias = Input(shape=[1])
        #Ajout des couches
        Conv2D_1 = Convolution2D(
            batch_input_shape=(self.BATCH_SIZE, state_size[2], state_size[1], state_size[0]),  
            filters=Filter2D_1,
            kernel_size=Kernel2D_1,     
            strides=1,
            padding='valid',              
            data_format='channels_first',     
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer=regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(State)
        Conv2D_2 = Convolution2D(
            batch_input_shape= (self.BATCH_SIZE, Filter2D_1, state_size[1]-Kernel2D_1[0]+1, state_size[0]),
            filters=Filter2D_2,
            kernel_size=Kernel2D_2,     
            strides=1,
            padding='valid',                   
            data_format='channels_first',      
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer= regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(Conv2D_1)
        L1 = Permute((3,2,1))(Conv2D_2)
        L2 = TimeDistributed(LSTM(units=Lstm,activation ='tanh',return_sequences = True),input_shape=[state_size[0],-1,Filter2D_2])(L1)
        L3 = TimeDistributed(LSTM(units=1, activation='linear', return_sequences=False), input_shape=[state_size[0],-1,Lstm])(L2)
        vote = Flatten()(L3)
        F1 = concatenate([vote,cash_bias],axis=1)
        action = Activation('softmax')(F1)
        model = Model(inputs=[State, last_action, cash_bias], outputs=action)
        return model, model.trainable_weights, State, last_action, cash_bias, vote
    def train(self,states,last_actions,futur_prices,cash_bias):
        # entraine le model
        self.sess.run(self.optimizer,
        feed_dict={
            self.state : states,
            self.last_action : last_actions,
            self.futur_price : futur_prices,
            self.cash_bias : cash_bias})
        