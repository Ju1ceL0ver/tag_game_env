import math
import matplotlib.pyplot as plt
import numpy as np
import io

class Player:
  def __init__(self, speed):
    self.speed=speed
    self.positions=[(0,0)]
    self.rewards=[]
    self.x=0
    self.y=0

  def move(self, degree,h,w):

    radians = math.radians(degree)

    delta_x = self.speed * math.cos(radians)
    delta_y = self.speed * math.sin(radians)

    if delta_x+self.x>w or delta_x+self.x<w:
      delta_x=0
    if delta_y+self.y>h or delta_y+self.y<h:
      delta_y=0

    self.x += delta_x
    self.y += delta_y
    self.positions.append((self.x,self.y))
  
class EnvTag:

  def __init__(self,p1,p2,w=100,h=100,near=0.5):

    self.p1=p1
    self.p2=p2
    self.reward=None
    self.running=True
    self.h=h
    self.w=w
    self.near=near

  def calculate_reward(self):

    x1=self.p1.x
    y1=self.p1.y
    x2=self.p2.x
    y2=self.p2.y

    reward=((x1-x2)**2+(y1-y2)**2)**0.5

    return reward
    
  def step(self,action1,action2):

    self.p1.move(action1)
    self.p2.move(action2)
    reward=self.calculate_reward()
    self.p1.rewards.append(reward)
    self.p2.rewards.append(reward)
    if reward<=self.near:
      self.running=False
    return reward

  def reset(self):
    self.p1.positions=[(0,0)]
    self.p2.positions=[(0,0)]
    self.p1.rewards=[]
    self.p2.rewards=[]
    self.running=True
    self.p1.x=0
    self.p1.y=0
    self.p2.x=0
    self.p2.y=0

  def draw(self):
    plt.figure(figsize=(3, 3))
    plt.xlim(self.w)
    plt.ylim(self.h)
    plt.scatter([p1.x],[p1.y],color='r',marker='s',s=self.h)
    plt.scatter([p2.x],[p2.y],color='b',marker='H',s=self.h)
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


    image = image.resize((300, 300)).convert('L')
    plt.close()

    return image
