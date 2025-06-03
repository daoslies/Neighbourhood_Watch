### Adapted from https://github.com/philtabor/Youtube-Code-Repository/tree/master

# RecordingOffice, SelectCommittee,  NeighbourhoodWatch,  TheCouncil

### LSTM edit

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from pathlib import Path

class RecordingOffice:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        #print(np.array(self.actions))
        #print(np.array(self.probs))
        #print(np.array(self.vals))
        #print(np.array(self.rewards))
        #print(np.array(self.dones))


        # 2 tensors land/resource, can't be arrayed just yet
        return  self.states,\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_records_office(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_records_office(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"#torch.device()
# 2

class SelectCommittee(nn.Module):
  def __init__(self, lr, model_name, chkpt_dir='Models/select_committee'):
      super(SelectCommittee, self).__init__()

      self.checkpoint_dir = chkpt_dir
      self.checkpoint_file = os.path.join(chkpt_dir, f'CouncilPPO_{model_name}')

      self.hidden_state = None

      self.flatten = nn.Flatten()

      self.land_conv = nn.Sequential(
          nn.Conv2d(4, 64, 2),
          nn.ReLU(),
          nn.Conv2d(64, 32, 2),
          nn.ReLU(),
          nn.Conv2d(32, 1, 2),
          nn.ReLU()
      )
      
      self.linear_resources = nn.Sequential(
          nn.Linear(5, 80),
          nn.ReLU(),
          nn.Linear(80, 160),
          nn.ReLU(),
          nn.Linear(160, 10)
      )

      self.linear_joiner = nn.Sequential(
        nn.Linear(59,200),
        nn.ReLU()
      )

      self.lstm_memory = nn.LSTM(
            input_size=200,  
            hidden_size=400,
            num_layers=2,
            batch_first=True
        )

      self.linear_decision = nn.Sequential(
          nn.Linear(400, 400),
          nn.ReLU(),
          nn.Linear(400, 600),
          nn.Softmax(dim=-1) #removed on the advice of steven.
      )

      self.optimizer = optim.Adam(self.parameters(), lr=lr)
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.to(self.device)


  def init_hidden(self, batch_size=1): 
    """
    For LSTM. I think batch_size is always 1? unless we're actually showing multiple steps at once? 
    
    But I think the point is we want the memory to be entirely in the cell and hidden states, we aren't giving it hard coded memory and actually providing it with previous steps. 

    """
    return (
        torch.zeros(2, batch_size, 400).to(self.device),  # h_0
        torch.zeros(2, batch_size, 400).to(self.device)   # c_0
    )


  def forward(self, land_tense, resource_tense, hidden_state=None):

      resource_embed = self.linear_resources(resource_tense).transpose(0,1)
      flat_land = self.flatten(self.land_conv(land_tense)).transpose(0,1)

      joined_embed = torch.concatenate((flat_land, resource_embed)).transpose(0,1)  ## Maybe swap to axis = 1 to remove transpose

      joined_embed = self.linear_joiner(joined_embed).unsqueeze(1)  # shape becomes [1, 1, 200]

      if self.hidden_state is None:
        self.hidden_state = self.init_hidden() # I think batch_size is always 1
      else:
        self.hidden_state = tuple(h.detach() for h in self.hidden_state)

      memory_embed, self.hidden_state = self.lstm_memory(joined_embed, self.hidden_state)   

      memory_embed = memory_embed.squeeze()

      decision_distribution = self.linear_decision(memory_embed)
      decision_distribution = Categorical(decision_distribution)

      return decision_distribution

  def save_checkpoint(self):
      Path((self.checkpoint_dir)).mkdir(parents=True, exist_ok=True)
      torch.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
      self.load_state_dict(torch.load(self.checkpoint_file))

  def check_for_checkpoint(self):
    if os.path.exists(self.checkpoint_file):
        return True
    else:
        return False



## Doubt
# 3
class NeighbourhoodWatch(nn.Module):
    def __init__(self, lr, model_name, chkpt_dir='Models/Crtic', fc1_dims=256, fc2_dims=256):
        super(NeighbourhoodWatch, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, f'NeighbourhoodWatchPPO_{model_name}')

        self.flatten = nn.Flatten()

        self.land_conv = nn.Sequential(
            nn.Conv2d(4, 64, 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 2),
            nn.ReLU()

        )
        self.linear_resources = nn.Sequential(
            nn.Linear(5, 80),
            nn.ReLU(),
            nn.Linear(80, 150),
            nn.ReLU(),
            nn.Linear(150, 10)
        )

        self.nbhd_watch = nn.Sequential(
            nn.Linear(59, 200),
            nn.ReLU(),
            nn.Linear(200, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, land_tense, resource_tense):

        land_embed = self.land_conv(land_tense)
        resource_embed = self.linear_resources(resource_tense).transpose(0,1)

        flat_land = self.flatten(land_embed).transpose(0,1)
        joined_embed = torch.concatenate((flat_land, resource_embed)).transpose(0,1)

        criticism = self.nbhd_watch(joined_embed)
        return criticism

    def save_checkpoint(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def check_for_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            return True
        else:
            return False


class TheCouncil:
    def __init__(self, model_name, gamma=0.99, lr=0.0000005, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda ## Generalised advantage estimation

        self.select_committee = SelectCommittee(lr, model_name)  ## Does the stuff
        self.nbhd_watch = NeighbourhoodWatch(lr, model_name)  ## Provides a losss..? for the select_committee
        self.records_office = RecordingOffice(batch_size)  ## d

    def remember(self, state, action, probs, vals, reward, done):
        self.records_office.store_records_office(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.select_committee.save_checkpoint()
        self.nbhd_watch.save_checkpoint()

    def load_models(self, load: bool):
        if load:
            print('... loading models ...')
            self.select_committee.load_checkpoint()
            self.nbhd_watch.load_checkpoint()

    def check_for_checkpoints(self):
        if self.select_committee.check_for_checkpoint() and self.nbhd_watch.check_for_checkpoint():
            return True
        else:
            return False

    def choose_action(self, land_tensors, resource_tensors):
        land_tensors = land_tensors.to(self.select_committee.device)
        resource_tensors = resource_tensors.to(self.select_committee.device)

        dist = self.select_committee(land_tensors, resource_tensors)
        value = self.nbhd_watch(land_tensors, resource_tensors)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    
    def learn(self):
        device = self.select_committee.device
    
        for _ in range(self.n_epochs):
            # 1) Sample rollout
            (
                state_arr,
                action_arr,
                old_logp_arr,
                value_arr,
                reward_arr,
                dones_arr,
                batches
            ) = self.records_office.generate_batches()
    
            N = len(reward_arr)
    
            # 2) Move to tensors
            rewards  = torch.tensor(reward_arr, dtype=torch.float32, device=device)
            values   = torch.tensor(value_arr, dtype=torch.float32, device=device)
            dones    = torch.tensor(dones_arr, dtype=torch.float32, device=device)
            old_logp = torch.tensor(old_logp_arr, dtype=torch.float32, device=device)
    
            # 3) Extend values by one zero so V_{T+1}=0 for GAE
            values_ext = torch.cat([values, torch.zeros(1, device=device)], dim=0)
    
            # 4) Compute GAE advantages (single backward pass)
            advantages = torch.zeros(N, dtype=torch.float32, device=device)
            gae = 0.0
            for t in reversed(range(N)):
                mask  = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * values_ext[t+1] * mask - values_ext[t]
                gae   = delta + self.gamma * self.gae_lambda * mask * gae
                advantages[t] = gae
    
            # 5) Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
            # 6) Compute critic targets
            returns = advantages + values
    
            # 7) Mini-batch updates
            for batch in batches:

                idx = batch   # e.g. idx = array([3,7,12,...])
            
                # 1) Gather the tuples for this mini-batch
                batch_states = [ state_arr[i] for i in idx ]
                # Now batch_states = [
                #   (land0, res0),
                #   (land1, res1),
                #   … 
                # ]
            
                # 2) Unzip into two lists
                land_list, res_list = zip(*batch_states)
                # land_list = [land0, land1, …], each of shape [10,10]
                # res_list  = [res0,  res1, …], each of shape [1,5]
            
                # 3) Stack them into batched tensors
                land_tensor = torch.stack(land_list, dim=0)          # -> [batch_size,10,10]
                resource_tensor  = torch.stack(res_list,  dim=0).squeeze(1)  # -> [batch_size,5]
            
                # 4) Move to device
                land_tensor = land_tensor.to(device=device, dtype=torch.float32)
                resource_tensor  = resource_tensor.to(device=device,  dtype=torch.float32)
            
                # 5) Now your actions, old_logp, adv, returns line up 1:1 with these
                actions  = torch.tensor(action_arr[idx], dtype=torch.int64, device=device)
                old_lps   = old_logp[idx]

                # --- forward pass ---
                dist       = self.select_committee(land_tensor, resource_tensor)
                value_pred = self.nbhd_watch(land_tensor, resource_tensor).squeeze(-1)
                new_logp   = dist.log_prob(actions)
    
                # --- clipped surrogate actor loss ---
                ratio       = torch.exp(new_logp - old_lps)
                adv_batch   = advantages[idx]
                clipped_rat = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                actor_loss  = -torch.min(ratio * adv_batch, clipped_rat * adv_batch).mean()
    
                # --- critic loss (MSE) ---
                ret_batch   = returns[idx]
                critic_loss = torch.nn.functional.mse_loss(value_pred, ret_batch)
    
                # --- combined optimization ---
                total_loss = actor_loss + 0.5 * critic_loss
                self.select_committee.optimizer.zero_grad()
                self.nbhd_watch.optimizer.zero_grad()
                total_loss.backward()
                self.select_committee.optimizer.step()
                self.nbhd_watch.optimizer.step()
    
        # 8) Clear buffer
        self.records_office.clear_records_office()
        
    """
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, criticicism,\
            reward_arr, dones_arr, batches = \
                    self.records_office.generate_batches()

            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(\
                      reward_arr[k] + self.gamma*criticicism[k+1]*\
                            (1-int(dones_arr[k])) - criticicism[k]\
                                     )

                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(self.select_committee.device)
            criticicism = torch.tensor(criticicism).to(self.select_committee.device)

            for batch in batches:

                batch = batch[0]

                land_tensor = torch.tensor(state_arr[batch][0].clone().detach() , dtype=torch.float).to(self.select_committee.device)
                resource_tensor = torch.tensor(state_arr[batch][1].clone().detach() , dtype=torch.float).to(self.select_committee.device)

                #states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.select_committee.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.select_committee.device)
                actions = torch.tensor(action_arr[batch]).to(self.select_committee.device)

                dist = self.select_committee(land_tensor, resource_tensor)
                nbhd_watch_value = self.nbhd_watch(land_tensor, resource_tensor)

                nbhd_watch_value = torch.squeeze(nbhd_watch_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]

                select_committee_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + criticicism[batch]
                nbhd_watch_loss = (returns-nbhd_watch_value)**2
                nbhd_watch_loss = nbhd_watch_loss.mean()

                total_loss = select_committee_loss + 0.5*nbhd_watch_loss
                self.select_committee.optimizer.zero_grad()
                self.nbhd_watch.optimizer.zero_grad()
                total_loss.backward()
                self.select_committee.optimizer.step()
                self.nbhd_watch.optimizer.step()

        self.records_office.clear_records_office()
    """

    def reset_memory(self):  ## For LSTM
        self.select_committee.hidden_state = None
