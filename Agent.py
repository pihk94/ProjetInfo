import tensorflow as tf
from tensorflow.keras.layers import Input,Convolution2D,Reshape,concatenate,multiply,Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.layers.core import Activation
from tensorflow.keras.models import Model
class Agent:
    def __init__(self,session,state_size,action_size,BATCH_SIZE,LR,reward,NN):
        self.sess = session
        self.state_size = state_size
        self.action_size = action_size
        #Hyperparamètres modifiable
        self.BATCH_SIZE = BATCH_SIZE
        self.DECAY_STEPS = 50000
        self.DECAY_RATE = 0.1
        self.GLOBAL_STEP = tf.Variable(0, trainable=False)
        self.LR = tf.compat.v1.train.exponential_decay(LR,self.GLOBAL_STEP,self.DECAY_STEPS,self.DECAY_RATE) # diminue le LR au cours du training, tous les 
        self.TRANSITION_FACTOR = 0.002
        tf.compat.v1.keras.backend.set_session(self.sess)
        #Initialisation du model
        if NN == "CNN":
            print('Construction du CNN')
            self.model,self.weights,self.state,self.last_action,self.cash_bias,self.prediction,self.test = self.CNN(state_size)
        #Choix de la fonction de récompense
        if reward == 'avg_log_cum_return':
            self.reward = self.avg_log_cum_return()
        #On prend l'optimiser ADAM qui est le plus souvent utilisé dans la littérature et le plus efficace
        self.optimizer = tf.compat.v1.train.AdamOptimizer(LR).minimize(self.reward,global_step =self.GLOBAL_STEP)
        self.futur_price = tf.compat.v1.placeholder(tf.float32,[None,state_size[0]+1])
        self.sess.run(tf.compat.v1.global_variables_initializer())
    def avg_log_cum_return(self):
        print("Reward : Rendements cumulés moyen logarithmique")
        self.cout_transaction = 1 - tf.reduce_sum(self.TRANSITION_FACTOR * tf.abs(self.model.output[:,:-1] - self.last_action), axis=1)
        return -tf.reduce_mean(tf.math.log(self.cout_transaction * tf.reduce_sum(self.model.output * self.futur_price,axis=1)))
    def CNN(self,state_size):
        #HYPERPARAMETRES
        Kernel2D_1 = (5,1)
        Filter2D_1 = 3 # Valeur souvent utilisé pour commencer
        Kernel2D_2 = (state_size[1] - Kernel2D_1[0]+1,1)
        Filter2D_2 = 20
        State = Input(shape=[state_size[2],state_size[1],state_size[0]])
        prediction = Input(shape=[state_size[0]])
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
        vote_p = multiply([vote,prediction])
        F1 = concatenate([vote_p,cash_bias],axis=1)
        #Préparation de l'output,i.e nos actions. On utilise la fonction softmax comme activation
        action = Activation('softmax')(F1)
        model = Model(inputs=[State,last_action,cash_bias,prediction],outputs=action)
        return model, model.trainable_weights, State, last_action,cash_bias,prediction,vote
        