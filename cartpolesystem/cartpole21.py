import gymnasium as gym
import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
"""
DEQUE : ACTUALLY IT IS A DATA-STRUCTURE USED TO EITHER REMOVE 
OR ADD VALUE HERE MY NETWORK STORES THE EXPERIENCES OF 
"""


#HEY I AM WRITING OFF SOME VALUES TO TUNABLE PARAMETERS
GAMMA=0.98
lr=1e-2
batch_size=32
memorysize=10000
epsilon_start=1.0
epsilon_stop=0.01
epsilon_decay=0.995
target_update_freq:int=10
num_episodes=100

# creating a neural network:
class DQN(nn.Module):
        def __init__(self,state_dimensions,action_dimensions):
                super().__init__()
                self.layer1=nn.Sequential(
                        nn.Linear(state_dimensions,out_features=128),
                        nn.LeakyReLU(),
                        nn.Linear(128,64),
                        nn.LeakyReLU(),
                        nn.Linear(64,action_dimensions)

                )
        def forward(self,x):
                return self.layer1(x)

# an agent should have a memory let us create it
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
                        t.tensor(actions,dtype=t.int64),
                        t.tensor(rewards,dtype=t.float32),
                        t.tensor(next_states,dtype=t.float32),
                        t.tensor(dones,dtype=t.float32)
                )
        def __len__(self):
                return len(self.buffer)

        


# here epsilon greeedy is used which is used for exploration and exploitation reade off

def select_action(state,policy_net,epsilon,action_dim):
        if random.random()<epsilon:
                return random.randrange(action_dim)
        else:
                with t.no_grad():
                        state_tensor=t.tensor(state,dtype=t.float32).unsqueeze(0)
                        q_values=policy_net(state_tensor)
                        return q_values.argmax().item()



# how a training step goes
def train_step(policy_net:DQN,target_net:DQN,buffer:ReplayBuffer,optimizer:t.optim):
        if(len(buffer)<batch_size):
                return
        states,actions,rewards,next_states,dones=buffer.sample(batch_size)
        q_values=policy_net(states).gather(1,actions.unsqueeze(1)).squeeze()
        next_q_values=target_net(next_states).max(1)[0]
        #this is bellman equation for agents to learn
        target_q_values=rewards+GAMMA*next_q_values*(1-dones)
        loss=nn.MSELoss()(q_values,target_q_values.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train():
        env=gym.make(id="CartPole-v1",render_mode="human")
        state_dim=env.observation_space.shape[0]
        action_dim=env.action_space.n
        policy_net=DQN(state_dimensions=state_dim,action_dimensions=action_dim)
        target_net=DQN(state_dimensions=state_dim,action_dimensions=action_dim)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer=optim.Adam(policy_net.parameters(),lr=lr)
        buffer=ReplayBuffer(memorysize)
        epsilon=epsilon_start
        rewards_history=[]
        for episode in range(num_episodes):
                state,_=env.reset()
                total_reward=0
                while True:
                        env.render()
                        action=select_action(state,policy_net,epsilon,action_dim)
                        next_state,reward,terminated,truncated,_=env.step(action)
                        done=terminated or truncated
                        buffer.push(state,action,reward,next_state,done)
                        state=next_state
                        total_reward+=reward
                        train_step(policy_net,target_net,buffer,optimizer)
                        if done:
                                break
                epsilon=max(epsilon_stop,epsilon*epsilon_decay)
                rewards_history.append(total_reward)
                if episode%target_update_freq==0:
                        target_net.load_state_dict(policy_net.state_dict())
                print(f"Episode {episode},Reward:{total_reward} || Epsilon : {epsilon:.3f}")
        env.close()
        plt.plot(rewards_history)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN Training Progress ")
        plt.show()


if __name__=="__main__":
        train()




