# i am writing imports which will be necessary for the implementation
#running this in kaggle will work better
import gymnasium as gym
import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random 
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd


# i am creating a policy network you can change activations and you can try 
class DQN(nn.Module):
        """This is a basic neural network with nonlinear activations """
        def __init__(self,state_dim,action_dim):
                super().__init__()
                
                self.net=nn.Sequential(
                        nn.Linear(state_dim,128),
                        nn.LeakyReLU(),
                        nn.Linear(128,128),
                        nn.LeakyReLU(),
                        nn.Linear(128,action_dim)
                )
        def forward(self,x):
                return self.net(x)
        
# we need agentic memory we use deque to store new experiences that could be used by agent

class ReplayBuffer:
        def __init__(self,capacity):
                self.buffer=deque(maxlen=capacity)
        def push(self,state,action,reward,next_state,done):
                self.buffer.append((state,action,reward,next_state,done))
        def sample(self,batch_size):
                batch=random.sample(self.buffer,batch_size)
                states,actions,rewards,next_states,dones=zip(*batch)
                return (
                        t.tensor(states,dtype=t.float32),
                        t.tensor(actions,dtype=t.int64).unsqueeze(1),
                        t.tensor(rewards,dtype=t.float32).unsqueeze(1),
                        t.tensor(next_states,dtype=t.float32),
                        t.tensor(states,dtype=t.float32).unsqueeze(1),

                )
        
        def __len__(self):
                return len(self.buffer)
        

#hey here 
def select_action(state,epsilon,policy_net,action_dim):
        if random.random()<epsilon:
                return random.randint(0,action_dim-1)
        else:
                state_tensor=t.tensor(state,dtype=t.float32).unsqueeze(0)
                with t.no_grad():
                        q_values=policy_net(state_tensor)
                return q_values.argmax().item()
        

# we are going to define train_step 
def train_step(policy_net,target_net,memory,optimizer,batch_size,gamma):
        if(len(memory)<batch_size):
                return
        #above statement let initial exploration
        states,actions,rewards,next_states,dones=memory.sample(batch_size)
        #it is expecting return 
        q_values=policy_net(states).gather(1,actions)
        with t.no_grad():
                #it tells what happens in next_state 
                next_q_values=target_net(next_states).max(1)[0].unsqueeze(1)
                target_q_values=rewards+gamma*next_q_values*(1-dones)

        loss=nn.functional.mse_loss(q_values,target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



env=gym.make("LunarLander-v3",render_mode="rgb_array")
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n

policy_net=DQN(state_dim,action_dim)
target_net=DQN(state_dim,action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer=optim.Adam(policy_net.parameters(),lr=1e-3)
memory=ReplayBuffer(capacity=10000)
epsilon=1.0
epsilon_stop=0.01
epsilon_decay=0.995
gamma=0.99
batch_size=64
num_episodes=500
episode_rewards=[]

for episode in range(num_episodes):
        state,_=env.reset()
        total_reward=0
        for t in range(1000):
                action=select_action(state,epsilon,policy_net,action_dim)
                next_state,reward,terminated,truncated,_=env.step(action)
                done = terminated or truncated
                memory.push(state,action,reward,next_state,done)
                state=next_state
                total_reward+=reward

                train_step(policy_net,target_net,memory,optimizer,batch_size,gamma)
                if done:
                        break
        episode_rewards.append(total_reward)
        epsilon=max(epsilon_stop,epsilon_stop*epsilon_decay)

        if episode%10==0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode : {episode} || Reward : {total_reward:.2f} || Epsilon : {epsilon:.3f}")
