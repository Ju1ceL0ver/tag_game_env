import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import math

class ActorCriticTag(nn.Module):
    def __init__(self):
        super(ActorCriticTag, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(32),
            nn.Flatten()
        )
        self.fc_common = nn.Linear(30_752, 128)
        
        self.actor = nn.Linear(128, 1)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.fc_common(x))
        action = torch.sigmoid(self.actor(x)) * 365
        value = self.critic(x)
        return action, value
    
class Player:

  def __init__(self,speed,model,limit_x,limit_y,role=1):

    self.limit_x=limit_x
    self.limit_y=limit_y
    self.speed=speed
    self.model=model
    self.role=role
    self.actions=[]
    self.x=torch.rand(1).item()*self.limit_x
    self.y=torch.rand(1).item()*self.limit_y
    self.positions=[(self.x,self.y)]

  
  def move(self,degree):

    radians = math.radians(degree)

    delta_x = self.speed * math.cos(radians)
    delta_y = self.speed * math.sin(radians)

    if delta_x+self.x<0 or delta_x+self.x>self.limit_x:
      delta_x=0
    if delta_y+self.y<0 or delta_y+self.y>self.limit_y:
      delta_y=0

    self.x += delta_x
    self.y += delta_y
    self.positions.append((self.x,self.y))
    self.actions.append(degree)

  def train(self):
    pass
  
class Point:
  def __init__(self,x,y):
    self.x=x
    self.y=y

class EnvTag:

  def __init__(self,point,player,limit_x=100,limit_y=100,near=0.5,limit_steps=400):


    self.point=point
    self.player=player
    self.limit_x=limit_x
    self.limit_y=limit_y
    self.near=near
    self.limit_steps=limit_steps


    self.steps=0
    self.predicted_rewards=[]
    self.actual_rewards=[]
    self.predicted_actions=[]

    self.running=True

  def calculate_reward(self):

    x1=self.point.x
    y1=self.point.y
    x2=self.player.x
    y2=self.player.y

    reward=((x1-x2)**2+(y1-y2)**2)**0.5

    return reward * self.player.role

  
  def draw(self):

    plt.figure(figsize=(3, 3))
    plt.xlim(self.limit_x)
    plt.ylim(self.limit_y)
    plt.scatter([self.point.x],[self.point.y],color='r',marker='s',s=(self.limit_x+self.limit_y)/2)
    plt.scatter([self.player.x],[self.player.y],color='b',marker='H',s=(self.limit_x+self.limit_y)/2)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.gcf().canvas.draw()
    

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', pad_inches=0, dpi=100)
    buffer.seek(0)
    image_data = np.frombuffer(buffer.read(), dtype=np.uint8)
    buffer.close()
    image=Image.open(io.BytesIO(image_data))
    image = image.resize((128,128)).convert('L')
    plt.close()

    return torch.tensor(np.array(image),dtype=torch.float32).unsqueeze(0).unsqueeze(0)

  
  def step(self):

    img=self.draw()
    action,value=self.player.model(img)
    self.predicted_actions.append(action.item())
    self.predicted_rewards.append(value.item())
    self.actual_rewards.append(self.calculate_reward())
    self.player.move(action.item())
    self.steps+=1

  def check_ending(self):
    if abs(self.calculate_reward())<=self.near:
      self.running=False
      return True
    if self.steps>self.limit_steps:
      self.running=False
      return True
    return False
  
  def reset(self):
    self.steps=0
    self.predicted_rewards=[]
    self.actual_rewards=[]
    self.predicted_actions=[]
    self.running=True
  
  def play_episode(self):
    self.reset()
    while self.running:
      self.step()
      self.check_ending()
  