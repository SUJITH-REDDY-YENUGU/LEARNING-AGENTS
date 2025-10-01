# REQUIRED IMPORTS

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from functools import cache
# DQN Network
class DQN(nn.Module):
                def __init__(self, state_dim,action_dim):
                        super().__init__()
                        self.layer1=nn.Sequential(
                                nn.Linear(state_dim,128),
                                nn.LeakyReLU(),
                                nn.Linear(128,action_dim)
                        )
                def forward(self,x):
                        return self.layer1(x)
                @cache
                def __len__(self):
                        return sum([i.numel() for i in self.parameters()])
                
# we re defining memory

class ReplayBuffer:
        def __init__(self,capacity=10000):
                self.buffer=deque(maxlen=capacity)
        def push(self,state,action,reward,next_state,done):
                self.buffer.append((state,action,reward,next_state,done))
        def sample(self,batch_size):
                batch=random.sample(self.buffer,batch_size)
                states,actions,rewards,next_states,dones=zip(*batch)
                return(
                        t.tensor(states,dtype=t.float32),
                        t.tensor(actions),
                        t.tensor(rewards,dtype=t.float32),
                        t.tensor(next_states,dtype=t.float32),
                        t.tensor(dones,dtype=t.float32),
                )
        def __len__(self):
                return len(self.buffer)

#action selection
def select_action(state,action_dim,epsilon,policy_net):
        if random.random()<epsilon:
                return random.randint(0,action_dim-1)
        else:
                with t.no_grad():
                        state_tensor=t.tensor(state,dtype=t.float32)
                        q_values=policy_net(state_tensor)
                        return q_values.argmax().item()
                

def train_step(policy_net,target_net,buffer,optimizer,batch_size,gamma):
        if(len(buffer))<batch_size:
                return
        states,actions,rewards,next_states,dones=buffer.sample(batch_size)
        q_values=policy_net(states).gather(1,actions.unsqueeze(1)).squeeze()
        next_q_values=target_net(next_states).max(1)[0]
        expected_q_values=rewards+gamma*next_q_values*(1-dones)
        loss=nn.functional.mse_loss(q_values,expected_q_values.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#visualizing episdoe frames
def show_episode(frames,delay=0.05):
        import time
        plt.ion()
        fig,ax=plt.subplots()
        for frame in frames:
                ax.clear()
                ax.imshow(frame)
                ax.axis("off")
                plt.draw()
                plt.pause(delay)
        plt.ioff()
        plt.show()


#trainging loop
def train_dqn():
        env=gym.make("MountainCar-v0",render_mode="rgb_array")
        state_dim=env.observation_space.shape[0]
        action_dim=env.action_space.n
        policy_net=DQN(state_dim,action_dim)
        target_net=DQN(state_dim,action_dim)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer=optim.Adam(policy_net.parameters(),lr=1e-3)
        buffer=ReplayBuffer()
        gamma=0.99 # discount factor of future rewards
        epsilon=1.00
        epsilon_decay=0.995
        min_epsilon=0.05
        batch_size=64
        target_update_freq=10
        reward_log=[]
        render_episodes=[0,50,100,150]
        render_frames=[]
        for episode in range(200):
                state,_=env.reset()
                total_reward=0
                episode_frames=[]
                for t in range(200):
                 if episode in render_episodes:
                        frame=env.render()
                        episode_frames.append(frame)
                 action=select_action(state=state,policy_net=policy_net,epsilon=epsilon,action_dim=action_dim)
                 next_state,reward,terminated,truncated,_=env.step(action)
                 buffer.push(state=state,action=action,reward=reward,next_state=next_state,done=terminated or truncated)
                 train_step(policy_net=policy_net,target_net=target_net,buffer=buffer,optimizer=optimizer,batch_size=batch_size,gamma=gamma)
                 state=next_state
                 total_reward+=reward
                 if terminated or truncated:
                        break
                if episode in render_episodes:
                 render_frames.append(episode_frames)
                reward_log.append(total_reward)
                epsilon = max(min_epsilon, epsilon * epsilon_decay)
                if episode % target_update_freq == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                print(f"Episode {episode} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

        env.close()

    # Save rewards to DataFrame
        df = pd.DataFrame({'Episode': range(len(reward_log)), 'Reward': reward_log})
        df.to_csv("mountaincar_rewards.csv", index=False)

    # Plot reward trend
        plt.plot(df['Episode'], df['Reward'])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('MountainCar DQN Training Progress')
        plt.grid(True)
        
        
        plt.show()
        for i, frames in enumerate(render_frames):
                print(f"Showing episode {render_episodes[i]}")
                show_episode(frames)
        show_episode(render_frames[0])

if __name__=="__main__":
        train_dqn()

