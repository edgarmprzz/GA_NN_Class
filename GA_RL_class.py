import random
import numpy as np
from configparser import ConfigParser
from keras.layers import Input, Dense, Embedding, LSTM, TimeDistributed
import keras.layers
from keras.models import Model
import csv
from keras.optimizers import Adam
from keras.models import load_model
import json
from keras.models import model_from_json

train_lables=[]
train_samples=[]

with open('x_data.csv','r') as file:
    reader=csv.reader(file)

    for i in reader:
        train_samples.append(i)

with open('y_data.csv','r') as file:
    reader=csv.reader(file)

    for i in reader:
        train_lables.append(i)


train_samples=np.array(train_samples)
train_lables=np.array(train_lables)
train_samples=np.delete(train_samples,0,axis=0)
train_lables=np.delete(train_lables,0,axis=0)
temp_train_data=[]
temp_label_data=[]
count=0
for i in range(0,943):
    for j in range(0,2):
        count=count+1
        if(count<=943):
            temp_train_data.append(train_samples[i,j])
        else:
            break
train_samples=np.array(temp_train_data)
train_lables=np.array(train_lables)
train_lables=train_lables.astype('float64')


class GA_RL:
    def __init__(self,name):
        self.layers=[]
        self.act_func=[]
        self.optmizer=[]
        self.loss=[]
        self.name=name
        self.generation=0
        self.input_layer=1
        self.output_layer=1


    def activation_function(self):
        activation_functions=['relu','sigmoid','tanh','exponential','hard_sigmoid','softmax']
        act_fun=random.randint(0,5)
        self.act_func.append(activation_functions[act_fun])

    def optmizers(self):
        optmizer_functions=['rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'Nadam' ]
        opt_fun=random.randint(0,5)
        self.optmizer.append(optmizer_functions[opt_fun])

    def loss_func(self):
        loss_functions=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh']
        loss_fun=random.randint(0,7)
        self.loss.append(loss_functions[loss_fun])

    def generation_level(self):
        self.generation+=1


    def Layer_generator(self):
        self.layers.append(random.randint(0,100))

    def agent_structure_info(self):
        print(self.__dict__)

    # def model_concatenator(self,parent1,parent2):


    def Agent_generator(self):
        self.Layer_generator()
        self.activation_function()
        self.optmizers()
        self.loss_func()
        input=Input(shape=(self.input_layer,))
        layer=Dense(self.layers[self.generation],activation=self.act_func[self.generation])(input)
        output=Dense(2,activation='softmax')(layer)

        model=Model(inputs=input, outputs=output)
        model.compile(optimizer=self.optmizer[self.generation],
                        loss=self.loss[self.generation],
                        metrics=['accuracy'])
        model.fit(train_samples,train_lables,validation_split=0.1,batch_size=10,epochs=25,shuffle='True', verbose=1)
        model.save(self.name+'.h5')


    def load_agents_model(self):
        model=load_model(self.name+'.h5')
        model.summary()


    # def model_fit(self):





agent1=GA_RL(name ="a1")
agent2=GA_RL(name ="a2")

agent1.Agent_generator()
agent2.Agent_generator()


agent1.agent_structure_info()
agent2.agent_structure_info()

agent1.load_agents_model()
agent2.load_agents_model()
