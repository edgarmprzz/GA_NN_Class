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

    def model_concatenator(self,agent1,agent2):
        parent1=load_model(agent1.name+'.h5')
        parent2=load_model(agent2.name+'.h5')
        parent1.layers.pop()
        parent2.layers.pop()
        parent1.summary()
        parent2.summary()
        # temp=GA_RL(name="temp")
        # temp.Agent_generator()
        new_parent=Model(inputs=[parent1.input, parent2.input], outputs=[parent1.output, parent2.output])
        new_parent.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])
        parent1_weights=parent1.get_weights()
        parent2_wights=parent2.get_weights()
        new_parent_weights=parent1_weights+parent2_wights
        new_parent.set_weights(new_parent_weights)
        new_parent.summary()
        # agents_concatenate=keras.layers.concatenate([parent1.output, parent2.output])
        # temp=GA_RL(name="temp")
        # temp.Layer_generator()
        # temp.activation_function()
        # temp.optmizers()
        # temp.loss_func()
        # input=parent1.input
        # transitional_layer=agents_concatenate(input)
        # layer=Dense(temp.layers[temp.generation],activation=temp.act_func[temp.generation],name=temp.name+'_'+'hidden_'+str(self.generation))(transitioanl_layer)
        # output=Dense(2, activation='softmax', name='output')(layer)

        # parent1.save(agent1.name+'_'+'transitional_'+str(agent1.generation)+'.h5')
        # parent2.save(agent2.name+'_'+'transitional_'+str(agent2.generation)+'.h5')

    # def transitioanl_layer(self,agent1,agent2):
    #     parent1=load_model(agent1.name+'_'+'transitional_'+str(agent1.generation)+'.h5')
    #     parent2=load_model(agent2.name+'_'+'transitional_'+str(agent2.generation)+'.h5')
    #     parent1.summary()

    def Agent_generator(self):
        self.Layer_generator()
        self.activation_function()
        self.optmizers()
        self.loss_func()




    def Agent_generator_model(self):
        self.Layer_generator()
        self.activation_function()
        self.optmizers()
        self.loss_func()
        input=Input(shape=(self.input_layer,), name='input_'+self.name)
        layer=Dense(self.layers[self.generation],activation=self.act_func[self.generation],name=self.name+'_'+'hidden_'+str(self.generation))(input)
        output=Dense(2,activation='softmax',name='output_'+self.name)(layer)

        model=Model(inputs=input, outputs=output)
        model.compile(optimizer=self.optmizer[self.generation],
                        loss=self.loss[self.generation],
                        metrics=['accuracy'])
        model.fit(train_samples,train_lables,validation_split=0.1,batch_size=10,epochs=25,shuffle='True', verbose=1)
        model.save(self.name+'.h5')


    def view_agents_model(self):
        model=load_model(self.name+'.h5')
        model.summary()


    # def agent_fitness(self):





agent1=GA_RL(name ="a1")
agent2=GA_RL(name ="a2")

agent1.Agent_generator_model()
agent2.Agent_generator_model()


# agent1.agent_structure_info()
# agent2.agent_structure_info()

# agent1.view_agents_model()
# agent2.view_agents_model()

agent1.model_concatenator(agent1,agent2)
# agent1.view_agents_model()
