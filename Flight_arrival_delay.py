#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
from torch import nn,optim
import torch.nn.functional as F
import datetime
from torch.utils import data
from random import sample,seed


# In[2]:


flights = pd.read_csv("./data/flights.csv",low_memory=False)
airlines = pd.read_csv("./data/airlines.csv")
airports = pd.read_csv("./data/airports.csv")


# In[3]:


airports = {ch:i for i,ch in enumerate(flights.ORIGIN_AIRPORT.unique())}
airlines = {ch:i for i,ch in enumerate(airlines.IATA_CODE)}
tail_nums = {ch:i for i,ch in enumerate(flights.TAIL_NUMBER.unique())}
airports['10666'] =len(airports)

def one_hot_encode(size,val):
    a = np.zeros((size,),dtype=int)
    a[(val-1)] = 1
    return a
flights['DATE'] = pd.to_datetime(flights[['YEAR','MONTH', 'DAY']])
variables_to_remove = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF', 'YEAR', 
                       'DATE', 'AIR_SYSTEM_DELAY',
                       'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
                       'WEATHER_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
                       'FLIGHT_NUMBER', 'AIR_TIME']
flights.drop(variables_to_remove, axis = 1, inplace = True)
flights = flights[['AIRLINE','TAIL_NUMBER','DAY', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY',
        'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
        'SCHEDULED_TIME', 'ELAPSED_TIME','MONTH']]
flights['AIRLINE'] = flights['AIRLINE'].apply(lambda x: airlines[x])
flights['ORGIN_AIRPORT_V'] = flights['ORIGIN_AIRPORT'].apply(lambda x: airports[x])
flights['DESTINATION__AIRPORT_V'] = flights['DESTINATION_AIRPORT'].apply(lambda x: airports[x])
flights['TAIL_NUMBER'] = flights['TAIL_NUMBER'].apply(lambda x: tail_nums[x])
# flights['DAY_OF_WEEK'] = flights['DAY_OF_WEEK'].apply(lambda x: one_hot_encode(7,x))
flights['DAY'] = flights['DAY'].apply(lambda x: one_hot_encode(31,x))
flights['MONTH'] = flights['MONTH'].apply(lambda x: one_hot_encode(12,x))
flights = flights[['AIRLINE','TAIL_NUMBER','ORGIN_AIRPORT_V',"DESTINATION__AIRPORT_V",'ARRIVAL_DELAY','DEPARTURE_DELAY','DAY','MONTH','SCHEDULED_ARRIVAL']]
flights.dropna(inplace =True)
flights = flights.reset_index()
# flights = flights[['AIRLINE','ORGIN_AIRPORT_V',"DESTINATION__AIRPORT_V",'ARRIVAL_DELAY','DAY_OF_WEEK','DEPARTURE_DELAY','DAY','MONTH','SCHEDULED_ARRIVAL']]
flights.head()


# In[4]:


class Generator(data.Dataset):
    def __init__(self,flights):
        self.y = flights.ARRIVAL_DELAY.to_list()
        self.x = {
            'TAIL_NUMBER':flights.TAIL_NUMBER.to_list(),
            'AIRLINE': flights.AIRLINE.to_list(),
            'DAY': flights.DAY.to_list(),
            'MONTH': flights.MONTH.to_list(),
            'SCHEDULED_ARRIVAL': flights.SCHEDULED_ARRIVAL.to_list(),
            'ORGIN_AIRPORT': flights.ORGIN_AIRPORT_V.to_list(),
            'DESTINATION__AIRPORT': flights.DESTINATION__AIRPORT_V.to_list()
        }
    def __len__(self):
        return len(self.y)
    def __getitem__(self,index):
        y = torch.tensor(self.y[index],dtype=torch.float)
        x = {
            'TAIL_NUMBER':torch.tensor(self.x['TAIL_NUMBER'][index],dtype=torch.float),
            'AIRLINE': torch.tensor(self.x['AIRLINE'][index],dtype=torch.float),
            'DAY': torch.tensor(self.x['DAY'][index],dtype=torch.float),
            'MONTH': torch.tensor(self.x['MONTH'][index],dtype=torch.float),
            'SCHEDULED_ARRIVAL': torch.tensor(self.x['SCHEDULED_ARRIVAL'][index],dtype=torch.float),
            'ORGIN_AIRPORT': torch.tensor(self.x['ORGIN_AIRPORT'][index],dtype=torch.float),
            'DESTINATION__AIRPORT': torch.tensor(self.x['DESTINATION__AIRPORT'][index],dtype=torch.float)
        }
        return x,y


# In[5]:


seed(1234)
indexes = flights.index
indexes = [i for i in indexes]
train_index = sample(indexes,round(len(indexes)*0.75))


# In[6]:


# %%timeit valid_index =[i for i in indexes]
# for i in indexes:
#     if i not in train_index:
#         valid_index.append(i)
params = {'batch_size': 1024,
          'shuffle': True,
          'num_workers': 6}


# In[7]:


train = flights.loc[train_index,:]


# In[8]:


train_gen = Generator(train)
train_loader = data.DataLoader(train_gen,**params)


# In[9]:


class Model(nn.Module):
    def __init__(self,drop=0.25):
        super(Model,self).__init__()
        self.input = 48
        self.drop_p = drop
        self.layer1 = nn.Linear(self.input,1024)
        self.layer2 = nn.Linear(1024,1024)
        self.layer3 = nn.Linear(1024,1)
        self.dropout = nn.Dropout(p=drop)
    
    def forward(self,x):
        batch_size = x['DAY'].shape[0]
        x['TAIL_NUMBER'] = x['TAIL_NUMBER'].reshape(batch_size,-1)
        x['AIRLINE'] = x['AIRLINE'].reshape(batch_size,-1)
        x['SCHEDULED_ARRIVAL'] = x['SCHEDULED_ARRIVAL'].reshape(batch_size,-1)
        x['ORGIN_AIRPORT'] = x['ORGIN_AIRPORT'].reshape(batch_size,-1)
        x['DESTINATION__AIRPORT'] = x['DESTINATION__AIRPORT'].reshape(batch_size,-1)
        if torch.cuda.is_available():
            x['TAIL_NUMBER'] = x['TAIL_NUMBER'].cuda()
            x['AIRLINE'] = x['AIRLINE'].cuda()
            x['SCHEDULED_ARRIVAL'] = x['SCHEDULED_ARRIVAL'].cuda()
            x['ORGIN_AIRPORT'] = x['ORGIN_AIRPORT'].cuda()
            x['DESTINATION__AIRPORT'] = x['DESTINATION__AIRPORT'].cuda()
            x['DAY'] = x['DAY'].cuda()
            x['MONTH'] = x['MONTH'].cuda()
        inp = torch.cat((x['DESTINATION__AIRPORT'],x['ORGIN_AIRPORT'],x['SCHEDULED_ARRIVAL'],x['AIRLINE'],x['TAIL_NUMBER'],
                        x['DAY'],x['MONTH']),dim=1)
        out = self.layer1(inp)
        out = self.dropout(torch.relu(out))
        out = self.layer2(out)
        out = self.dropout(torch.relu(out))
        out = self.layer3(out)
        
        return out


# In[10]:


model = Model()
lr = 0.0001
criterian = nn.SmoothL1Loss() 
optimizer = optim.Adam(model.parameters(),lr =lr)


# In[ ]:


epoch = 5
if torch.cuda.is_available():
    model.cuda()
for e in range(epoch):
    run_loss =0
    count = 0
    optimizer.zero_grad()
    for x,y in train_loader:
        if torch.cuda.is_available():
            y = y.cuda()
        y = y.reshape(y.shape[0],-1)
        out = model(x)
        loss = criterian(out,y)
        run_loss += loss.item()
        loss.backward()
        optimizer.step()
        count +=1
    else:
        run_loss = run_loss/count
        print('{} loss = {}'.format(e,run_loss))


# In[ ]:




