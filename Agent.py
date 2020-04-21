import tensorflow as tf

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
            self.reward = reward
        #On prend l'optimiser ADAM qui est le plus souvent utilisé dans la littérature et le plus efficace
        self.optimizer = tf.compat.v1.train.AdamOptimizer(LR).minimize(self.reward,global_step =self.GLOBAL_STEP)
        self.sess.run(tf.compat.v1.global_variables_initializer())
    def avg_log_cum_return(self):
        print("Reward : Rendements cumulés moyen logarithmique")
        self.cout_transaction = 1 - tf.reduce_sum(self.TRANSITION_FACTOR * tf.abs(self.model.output[:,:-1] - self.last_action), axis=1)
        return -tf.reduce_mean(tf.math.log(self.cout_transaction * tf.reduce_sum(self.model.output * self.futur_price,axis=1)))
    def CNN(self,state_size):
        #HYPERPARAMETRES
        pass