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
from keras import Sequential
from Simulator import Environment,GUI,RL,Utils
import rlcarsim
from configobj import ConfigObj

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

agents_list=[]


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
        self.model=Model()
        self.check_member(agents_list)

    def __del__(self):
        print('parent was destroyed ')

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

    def model_pop_output(self):
        self.generation_level()
        self.model.layers.pop()
        self.random_generator()
        new_parent=Model(inputs=self.model.input, outputs=self.model.layers[-1].output)
        new_parent.compile(optimizer=self.optmizer[self.generation],
                    loss=self.loss[self.generation],
                    metrics=['accuracy'])
        new_parent.set_weights(self.model.get_weights())
        self.model=new_parent

    # def transitional_layer(self,layer):
    #     temp=GA_RL(name="temp")
    #     temp.random_generator()
    #     input=Input(shape=(self.input_layer,), name='input_'+self.name)
    #     layer=layer(i
    #     temp.model.trainable=False
    #     new_model=Sequential()
    #     new_model.add(temp.model)
    #     new_model.summary()
    #     new_model.add(layer)
    #     new_model.compile(optimizer=self.optmizer[self.generation+1],
    #                 loss=self.loss[self.generation+1],
    #                 metrics=['accuracy'])
    #     self.model=new_model

    def mutate(self):
        rate= random.uniform(0,1)
        if (rate<0.5):
            new_model=Sequential()
            new_model.add(self.model)
            new_model.add(dropout(0.33))
            self.model=new_model
        else:
            pass

    def concatenate_models(self,agent):
        temp=GA_RL(name="child")
        temp.Agent_generator_model()
        temp.model.layers.pop()
        temp.model.layers.pop()
        self.model.layers[-1].output.trainable=False
        agent.model.layers[-1].output.trainable=False
        concat_layers=keras.layers.Concatenate(axis=1, name=self.name+'_concatenated_layer_'+str(self.generation-1))([self.model.layers[-1].output, agent.model.layers[-1].output])
        output=Dense(2,activation='softmax',name='output_'+self.name)(concat_layers)
        temp.model=Model(inputs=[self.model.input,agent.model.input], outputs=output)
        temp.model.compile(optimizer=self.optmizer[self.generation+1],
                        loss=self.loss[self.generation+1],
                        metrics=['accuracy'])
        return temp

    def random_generator(self):
        self.Layer_generator()
        self.activation_function()
        self.optmizers()
        self.loss_func()

    def Agent_generator_model(self):
        self.random_generator()
        input=Input(shape=(self.input_layer,), name='input_'+self.name)
        layer=Dense(self.layers[self.generation],activation=self.act_func[self.generation],name=self.name+'_'+'hidden_'+str(self.generation))(input)
        output=Dense(2,activation='softmax',name='output_'+self.name)(layer)

        self.model=Model(inputs=input, outputs=output)
        self.model.compile(optimizer=self.optmizer[self.generation],
                        loss=self.loss[self.generation],
                        metrics=['accuracy'])
        self.model.fit(train_samples,train_lables,validation_split=0.1,batch_size=10,epochs=25,shuffle='True', verbose=1)

    def view_agents_model(self):
        self.model.summary()

    def save_agents_model(self):
        self.model.save('/home/edgar/gen_alg/agents_networks/'+self.name+'_model_'+str(self.generation)+'.h5')

    def save_model_attributes(list):
        path='/home/edgar/gen_alg/agents_networks/Config_GA.ini'
        config=ConfigObj(path)
        config.file=path
        config["Network"]={}
        for obj in list:
            for layers in obj.layers:
                config["Network"]["layers"]=obj.layers[layers]
            for func in obj.act_func:
                config["Network"]["activation_functions"]=obj.act_func[func]
            for opt in obj.optmizer:
                config["Network"]["optimizer"]=obj.optmizer[opt]
            for loss in obj.loss:
                config["Network"]["loss"]=obj.loss[loss]
        config.write()
        return True

    def check_member(self,list):
        for members in agents_list:
            if self.name==members.name:
                break
            else:
                agents_list.append(self)

    def agent_structure_info(self):
        print(self.__dict__)

if(__name__=='__main__'):
    agent1=GA_RL(name ="agent_1")
    agent2=GA_RL(name ="agent_2")

    agent1.random_generator()
    agent2.random_generator()

    agent1.Agent_generator_model()
    agent2.Agent_generator_model()

    agent1.model_pop_output()
    agent2.model_pop_output()

    concatenation_a1=agent1.concatenate_models(agent2)
    # concatenation_a2=agent2.concatenate_models(agent1)
    #
    #
    agent1.view_agents_model()
    agent2.view_agents_model()
    #
    agent1.save_agents_model()
    agent2.save_agents_model()
    # concatenation_a1.view_agents_model()
    concatenation_a1.agent_structure_info()
    # concatenation_a2.view_agents_model()
    print(len(agents_list))
    print(save_model_attributes(agents_list))
