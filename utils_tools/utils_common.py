# -*- coding: utf-8 -*-
import random
import torch
from utils_tools.net import Actor, Critic, Common
import itertools
from skimage import color
from torch.autograd import Variable
from collections import deque
import copy
import time
import os

time_Feature = round(time.time())


class DDPG:
    def __init__(self, state_size, action_size, device_):
        self.t = 0
        self.max_Q = 0
        self.trainingLoss = 0
        self.train = True
        self.train_from_checkpoint = False

        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.device = device_

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        # if self.train and not self.train_from_checkpoint:
        #     self.epsilon = 1.0
        #     self.initial_epsilon = 1.0
        # else:
        #     self.epsilon = 0
        #     self.initial_epsilon = 0
        self.batch_size = 16
        self.train_start = 2000
        self.train_from_checkpoint_start = 3000
        self.tua = 1e-3
        # 初始化history存放参数 ！！！可以不使用，直接使用train_replay返回值做
        self.history_loss_actor = 0.1
        self.history_loss_critic = 0.1

        # Create replay memory using deque
        self.memory = deque(maxlen=32000)

        # Create main model and target model
        self.common_model = Common().to(self.device)
        self.common_target_model = Common().to(self.device)

        self.actor_model = Actor().to(self.device)
        self.actor_target_model = Actor().to(self.device)

        self.critic_model = Critic().to(self.device)
        self.critic_target_model = Critic().to(self.device)
        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.opt_actor = torch.optim.Adam(itertools.chain(self.common_model.parameters(),
                                                          self.actor_model.parameters()),
                                          lr=1e-4)
        self.opt_critic = torch.optim.Adam(itertools.chain(self.common_model.parameters(),
                                                           self.critic_model.parameters()),
                                           lr=1e-3, weight_decay=1e-2)
        # self.loss_actor = torch.nn.MSELoss()
        self.loss_critic = torch.nn.MSELoss()

        hard_update_target_model(self.common_model, self.common_target_model)
        hard_update_target_model(self.actor_model, self.actor_target_model)
        hard_update_target_model(self.critic_model, self.critic_target_model)

    @staticmethod
    def process_image(obs):
        obs = color.rgb2gray(obs)
        return obs

    # Get action from model using epsilon-greedy policy
    def get_action(self, Input, noise_added=None):
        self.common_model.eval()
        self.actor_model.eval()
        common = self.common_model(Input[0], Input[1])
        mu = self.actor_model(common)
        self.actor_model.train()
        self.common_model.train()
        if noise_added is not None:
            # noise_added = torch.from_numpy(noise_added.__call__()).to(self.device)
            noise_added = torch.from_numpy(noise_added.__call__()).to(self.device)
            mu += noise_added
            mu = torch.clamp(mu, min=self.action_size[0], max=self.action_size[1])
        return mu

    def replay_memory(self, state, v_ego, action, reward, next_state, nextV_ego, done):
        self.memory.append((state, v_ego, action, reward, next_state, nextV_ego, done, self.t))

    def DDPG_train_replay(self):
        if len(self.memory) < self.train_start:
            return
        elif self.train_from_checkpoint:
            if len(self.memory) < self.train_from_checkpoint_start:
                return
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        '''
        torch.float64对应torch.DoubleTensor
        torch.float32对应torch.FloatTensor
        '''
        # from_numpy会对numpy数据类型转换成双精度张量，而torch.Tensor不存在这种问题，torch.Tensor将数组转换成单精度张量
        state_t, v_ego_t, action_t, reward_t, state_t1, v_ego_t1, terminal, step = zip(*minibatch)
        state_t = Variable(torch.Tensor(state_t).squeeze().to(self.device))
        state_t1 = Variable(torch.Tensor(state_t1).squeeze().to(self.device))
        v_ego_t = Variable(torch.Tensor(v_ego_t).squeeze().to(self.device))
        v_ego_t1 = Variable(torch.Tensor(v_ego_t1).squeeze().to(self.device))
        reward_t = torch.Tensor(reward_t).reshape(-1, 1).to(self.device)
        action_t = torch.Tensor(action_t).reshape(-1, 1).to(self.device)

        self.opt_critic.zero_grad()
        feature_extraction_target = self.common_target_model(state_t1, v_ego_t1)
        action_target = self.actor_target_model(feature_extraction_target)

        td_target = reward_t + self.discount_factor * self.critic_target_model(feature_extraction_target, action_target.detach())

        feature_extraction = self.common_model(state_t, v_ego_t)
        critic_loss_cal = self.loss_critic(self.critic_model(feature_extraction, action_t), td_target.detach())
        critic_loss_cal.backward()
        self.opt_critic.step()
        self.history_loss_critic = critic_loss_cal.item()

        # 重置critic，actor优化器参数
        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        acotr_feature_extraction = self.common_model(state_t, v_ego_t)
        policy_actor = -self.critic_model(acotr_feature_extraction, self.actor_model(acotr_feature_extraction))
        policy_actor = policy_actor.mean()
        policy_actor.backward()
        self.opt_actor.step()
        self.history_loss_actor = policy_actor.item()

        soft_update_target_model(self.common_model, self.common_target_model, self.tua)
        soft_update_target_model(self.actor_model, self.actor_target_model, self.tua)
        soft_update_target_model(self.critic_model, self.critic_target_model, self.tua)
        return

    def load_model(self, name):
        checkpoints = torch.load(name)
        self.critic_model.load_state_dict(checkpoints['model_critic'])
        self.actor_model.load_state_dict(checkpoints['model_actor'])
        self.common_model.load_state_dict(checkpoints['model_common'])
        self.opt_critic.load_state_dict(checkpoints['optimizer_critic'])
        self.opt_actor.load_state_dict(checkpoints['optimizer_actor'])

    # Save the model which is under training
    def save_model(self, name):
        torch.save({'model_actor': self.actor_model.state_dict(),
                    'model_critic': self.critic_model.state_dict(),
                    'model_common': self.common_model.state_dict(),
                    'optimizer_actor': self.opt_actor.state_dict(),
                    'optimizer_critic': self.opt_critic.state_dict()}, name)


class PPO:
    def __init__(self, state_size, action_size, device_):
        self.t = 0
        self.max_Q = 0
        self.trainingLoss = 0
        self.train = True
        self.train_from_checkpoint = False

        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.device = device_

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        # if self.train and not self.train_from_checkpoint:
        #     self.epsilon = 1.0
        #     self.initial_epsilon = 1.0
        # else:
        #     self.epsilon = 0
        #     self.initial_epsilon = 0
        self.batch_size = 16
        self.train_start = 2000
        self.train_from_checkpoint_start = 3000
        self.tua = 1e-3
        # 初始化history存放参数 ！！！可以不使用，直接使用train_replay返回值做
        self.history_loss_actor = 0.1
        self.history_loss_critic = 0.1

        # Create replay memory using deque
        self.memory = deque(maxlen=32000)

        # Create main model and target model
        self.common_model = Common().to(self.device)
        self.actor_model = Actor().to(self.device)

        self.critic_model = Critic().to(self.device)
        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.opt_actor = torch.optim.Adam(itertools.chain(self.common_model.parameters(),
                                                          self.actor_model.parameters()),
                                          lr=1e-4)
        self.opt_critic = torch.optim.Adam(itertools.chain(self.common_model.parameters(),
                                                           self.critic_model.parameters()),
                                           lr=1e-3)
        # self.loss_actor = torch.nn.MSELoss()
        self.loss_critic = torch.nn.MSELoss()

    @staticmethod
    def process_image(obs):
        obs = color.rgb2gray(obs)
        return obs

    # Get action from model using epsilon-greedy policy
    def get_action(self, Input, noise_added=None):
        self.common_model.eval()
        self.actor_model.eval()
        common = self.common_model(Input[0], Input[1])
        mu = self.actor_model(common)
        self.actor_model.train()
        self.common_model.train()
        if noise_added is not None:
            # noise_added = torch.from_numpy(noise_added.__call__()).to(self.device)
            noise_added = torch.from_numpy(noise_added.__call__()).to(self.device)
            mu += noise_added
            mu = torch.clamp(mu, min=self.action_size[0], max=self.action_size[1])
        return mu

    def replay_memory(self, state, v_ego, action, reward, next_state, nextV_ego, done):
        self.memory.append((state, v_ego, action, reward, next_state, nextV_ego, done, self.t))

    def PPO_train_replay(self, ):
        pass


# target model硬更新
def hard_update_target_model(model, target_model):
    # 解决state_dict浅拷贝问题
    weight_model = copy.deepcopy(model.state_dict())
    target_model.load_state_dict(weight_model)


# target model软更新
def soft_update_target_model(model, target_model, t):
    for target_param, source_param in zip(target_model.parameters(),
                                          model.parameters()):
        target_param.data.copy_((1 - t) * target_param + t * source_param)


# 单目标斜对角坐标
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyxy2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2]
    y[:, 3] = x[:, 3]
    y = y.type(torch.IntTensor)
    return y


def print_out(file, text):
    file.write(text + '\n')
    file.flush()


def log_File_path(path):
    # date = str(dt.date.today()).split('-')
    # date_concat = date[1] + date[2]
    date_concat = time_Feature
    train_log_ = open(os.path.join(path, 'train_log_{}.txt'.format(date_concat)), 'w')
    test_log_ = open(os.path.join(path, 'test_log_{}.txt'.format(date_concat)), 'w')
    del date_concat
    return train_log_, test_log_


def Model_save_Dir(PATH, time):
    path_to_return = os.path.join(PATH, 'save_model_{}'.format(time)) + '/'
    if not os.path.exists(path_to_return):
        os.mkdir(path_to_return)
    return path_to_return


def reward_discrete_counter(counter, reward_value):
    if reward_value < 0.05:
        return counter + 1
    else:
        return 0