import numpy as np 
import random
class Buffer:
    def __init__(self,BUFFER_SIZE,SAMPLE_BIAIS):
        self.SAMPLE_BIAIS = SAMPLE_BIAIS
        self.BUFFER_SIZE = BUFFER_SIZE
        self.buffer = []
        self.total_experiences = 0
    def taille(self):
        #obtenir la taille du buffer
        return self.BUFFER_SIZE
    def nb_total_exp(self):
        #obtenir le nombre totale d'experiences
        return self.total_experiences
    def clear(self):
        #nettoyage du buffer rapide
        self.buffer = []
        self.total_experiences = 0
    def echantillon(self,size,SAMPLE_BIAIS):
        #echantillon aléatoire
        poids = np.array([SAMPLE_BIAIS**i for i in range(1,size+1)])
        return np.random.choice(size, 1, p =poids/sum(poids))[0]
    def getBatch(self, BATCH_SIZE):
        if self.total_experiences <= BATCH_SIZE:
            return self.buffer, self.total_experiences
        else:
            random_batch = self.echantillon(self.total_experiences-BATCH_SIZE, self.SAMPLE_BIAIS)
            return self.buffer[random_batch : random_batch + BATCH_SIZE], BATCH_SIZE
    def add(self, state, futur_price, last_action):
        #ajout des différentes valeurs à l'objet
        exp = (state,futur_price,last_action)
        if self.total_experiences  < self.BUFFER_SIZE:
            self.buffer.append(exp)
            self.total_experiences += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(exp)
