import base64
import copy
import datetime as dt
import io
import os
import random
import socket
import sys
import time
from collections import deque
import itertools
from utils_tools.net import Common, Actor, Critic
from utils_tools.ou_noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise, OUNoise
import numpy as np
import skimage
import torch
from torch.autograd import Variable

from PIL import Image, ImageDraw
from skimage import color
import cv2

np.set_printoptions(precision=4)

EPISODES = 5000
img_rows, img_cols = 80, 80
Distance_EF = 70
Return_Time = 3.5
Variance = 0.5
# Convert image into gray scale
# We stack 8 frames, 0.06*8 sec
img_channels = 4 
unity_Block_size = 65536
# PATH_MODEL = 'C:/dl_data/Python_Project/save_model/'
# PATH_LOG = 'C:/dl_data/Python_Project/train_log/'
CHECK_POINT_TRAIN_PATH = './save_Model/save_model_1631506899/save_model_398.h5'
PATH_MODEL = 'save_Model'
PATH_LOG = 'train_Log'
time_Feature = round(time.time())
random_index = np.random.permutation(img_channels)


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
        obs = skimage.color.rgb2gray(obs)
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
            mu = torch.clamp(mu, min=action_size[0], max=action_size[1])
        return mu

    def replay_memory(self, state, v_ego, action, reward, next_state, nextV_ego, done):
        self.memory.append((state, v_ego, action, reward, next_state, nextV_ego, done, self.t))

    def train_replay(self):
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
        v_ego_t1 = Variable(torch.Tensor(v_ego_t1).squeeze().to(device))
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
        self.model.load_state_dict(checkpoints['model'])
        self.optimizer.load_state_dict(checkpoints['optimizer'])

    # Save the model which is under training
    def save_model(self, name):
        torch.save({'model_actor': self.actor_model.state_dict(),
                    'model_critic': self.critic_model.state_dict(),
                    'model_common': self.common_model.state_dict(),
                    'optimizer_actor': self.opt_actor.state_dict(),
                    'optimizer_critic': self.opt_critic.state_dict()}, name)


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


def decode(revcData, v_ego_=0, v_lead_=0, force=0, episode_len=0, v_ego_copy_=0, v_lead_copy_=0):
    # received data processing
    revcList = str(revcData).split(',', 4)
    gap_ = float(revcList[0][2:])                    # distance between vehicles
    v_ego1_ = float(revcList[1])                      # speed of egoVehicle
    v_lead1_ = float(revcList[2])                     # speed of leadVehicle
    a_ego1_ = float(revcList[3])                      # acceleration of egoVehicle
    img = base64.b64decode(revcList[4])      # image from mainCamera
    image = Image.open(io.BytesIO(img))
    # image resize, 双线性插值
    images = np.array(image)

    anchor_box = yolo(images)
    # !!!后期可能会出现问题（问题具体出现在锚框不准确的问题）!!!
    # 后期优化方向，取置信度最大的锚框index作为锚框选取坐标
    # index = anchor_box.pred[0][0, 4]

    # 当前代码问题，yolo输出锚框为1时选取锚框问题，暂时解决通过取消squeeze
    # !!!锚框缺失问题!!!
    position = np.array(xyxy2xyxy(anchor_box.pred[0]), dtype='uint8')

    # pillow 支持
    # new_graph = Image.new(mode='1', size=(400, 400))
    if position.shape[0] != 0:
        ImageDraw.Draw(image).rectangle([(position[0, 0], position[0, 1]), (position[0, 2], position[0, 3])], fill='white', width=3)
    image = image.resize((80, 80), resample=Image.BILINEAR)
    image = np.array(image)

    '''
    opencv支持
    if position.shape[0] != 0:
        image = cv2.rectangle(image, (position[0, 0], position[0, 1]), (position[0, 2], position[0, 3]), (0, 255, 0), 2)
    # 更改双线性插值为区域插值，图像处理完效果好于双线性插值
    image = cv2.resize(image, (80, 80), interpolation=cv2.INTER_AREA)
    '''

    # 计算reward
    done_ = 0
    v_relative = v_ego1_ - v_lead1_
    action_relative = ((v_ego1_ - v_ego_) / 0.5) - ((v_lead1_ - v_lead_) / 0.5)
    action_best = (-(Distance_EF - float(gap_)) - v_relative * Return_Time) / (2 * Return_Time ** 2)

    reward = CalReward(action_relative, action_best)
    if 40 <= gap_ <= 60:
        pass
    elif 5 < gap_ < 40:
        reward = reward * ((1 / 35) * gap_ - (1 / 7))
    elif 60 <= gap_ <= 300:
        reward = reward * (- (1 / 240) * gap_ + (5 / 4))

    if gap_ <= 3 or gap_ >= 300:
        done_ = 1
        reward = -1.0
    elif episode_len > 480:
        done_ = 2
        # reward = CalReward(float(gap), float(v_ego), float(v_lead), force)
    return image, float(reward), done_, float(gap_), float(v_ego1_), float(v_lead1_), float(a_ego1_)


# 修改正态分布中的方差值，需要重新将正态分布归一化
def CalReward(action_relative_, action_best_):
    try:
        reward_recal = (np.exp(-(action_relative_ - action_best_) ** 2 / (2 * (Variance ** 2))))
    except FloatingPointError as e:
        reward_recal = 0
    return reward_recal


def reset():
    strr = str(3) + ',' + '0.0'
    sendDataLen = sock.sendto(strr.encode(), (remoteHost, remotePort))


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


def Recv_data_Format(byte_size, _done, v_ego=None, v_lead=None, action=None, episode_len=None, s_t=None, v_ego_t=None):
    if _done != 0: 
        revcData, (remoteHost, remotePort) = sock.recvfrom(byte_size)
        image, _, _, gap, v_ego, v_lead, a_ego = decode(revcData)
        x_t = agent.process_image(image)

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
        v_ego_t = np.array((v_ego, v_ego, v_ego, v_ego))

        # In Keras, need to reshape
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])      # 1*80*80*4
        v_ego_t = v_ego_t.reshape(1, v_ego_t.shape[0])      # 1*4
        return s_t, v_ego_t, v_ego, v_lead, remoteHost, remotePort
    else:
        revcData, (remoteHost, remotePort) = sock.recvfrom(byte_size)

        image, reward, done, gap, v_ego1, v_lead, a_ego1 = decode(revcData, v_ego, v_lead, action, episode_len)

        x_t1 = agent.process_image(image)
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])     # 1x1x80x80
        s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)    # 1x4x80x80
        v_ego_1 = np.array(v_ego1)
        v_ego_1 = np.expand_dims(v_ego_1, -1)
        v_ego_1 = np.expand_dims(v_ego_1, -1)
        v_ego_t1 = np.append(v_ego_1, v_ego_t[:, :3], axis=1)   # 1x4
        return reward, done, gap, v_ego1, v_lead, a_ego1, v_ego_1, s_t1, v_ego_t1


# def Send_data_Format(remoteHost, remotePort, onlyresetloc, s_t, v_ego_t):
def Send_data_Format(remoteHost, remotePort, s_t, v_ego_t, episode_len, UnityReset, action_noise=None):
    pred_time_pre = dt.datetime.now()
    episode_len = episode_len + 1            
    # Get action for the current state and go one step in environment
    s_t = torch.Tensor(s_t).to(device)
    v_ego_t = torch.Tensor(v_ego_t).to(device)
    force = agent.get_action([s_t, v_ego_t], action_noise)
    # 为了防止tanh输出达到边界值，导致神经元失活，对输出数据做clip固定在（-0.5， 0.5）之后再做重映射
    action_send = force.item()

    if UnityReset == 1: 
        strr = str(4) + ',' + str(action_send)
        UnityReset = 0
    else:
        strr = str(1) + ',' + str(action_send)
    
    sendDataLen = sock.sendto(strr.encode(), (remoteHost, remotePort))  # 0.06s later receive
    pred_time_end = dt.datetime.now()
    time_cost = pred_time_end - pred_time_pre
    return episode_len, force.item(), time_cost, UnityReset


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


if __name__ == "__main__":
    if not os.path.exists('./' + PATH_LOG):
        os.mkdir(os.path.join(os.getcwd().replace('\\', '/'), PATH_LOG))
    if not os.path.exists('./' + PATH_MODEL):
        os.mkdir(os.path.join(os.getcwd().replace('\\', '/'), PATH_MODEL))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 8001))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)
    action_size = (-1, 1)   # env.action_space.n # Steering and Throttle

    train_log, test_log = log_File_path(PATH_LOG)
    PATH_ = Model_save_Dir(PATH_MODEL, time_Feature)
    agent = DDPG(state_size, action_size, device)
    episodes = []

    # noise = OrnsteinUhlenbeckActionNoise(np.zeros(1), np.ones(1) * 0.2)
    noise = OrnsteinUhlenbeckActionNoise(np.zeros(1), np.ones(1) * 0.3)

    if not agent.train:
        print("Now we load the saved model")
        agent.load_model('C:/DRL_data/Python_Project/Enhence_Learning/save_Model/save_model_1627300305/save_model_398.h5')
    elif agent.train_from_checkpoint:
        agent.load_model(CHECK_POINT_TRAIN_PATH)
        print(f'Now we load checkpoints for continue training:  {CHECK_POINT_TRAIN_PATH.split("/")[-1]}')
    else:
        print('Thread Ready!!!')

    # initialize value
    done = 0
    counter_value = 0

    # 增加yolo目标检测算法支持
    torch.hub.set_dir('./')
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/nc1_car.pt').to(device)

    for e in range(EPISODES):      
        print("Episode: ", e)
        # ou noise重置
        noise.reset()

        if done == 2:
            print("new continued epicode!")
            done = 0
            UnityReset = 1
            episode_len = 0
        else:
            # 后期重置进入第一次recv
            print('done value:', done)
            print("new fresh episode!")
            done = 1
            s_t, v_ego_t, v_ego, v_lead, remoteHost, remotePort = Recv_data_Format(unity_Block_size, done)
            done = 0
            UnityReset = 0
            episode_len = 0

        while done == 0:
            start_time = time.time()
            if agent.t % 1000 == 0:
                rewardTot = []
            episode_len, action, time_cost, UnityReset = Send_data_Format(remoteHost, remotePort, s_t, v_ego_t, episode_len, UnityReset, noise)
            reward, done, gap, v_ego1, v_lead, a_ego1, v_ego_1, s_t1, v_ego_t1 = Recv_data_Format(unity_Block_size, done, v_ego, v_lead, action, episode_len, s_t, v_ego_t)
            counter_value = reward_discrete_counter(counter_value, reward)
            rewardTot.append(reward)
            start_count_time = int(round(time.time() * 1000))

            if agent.train:
                # s_t, v_ego_t, s_t1, v_ego_t1 = random_sample(s_t, v_ego_t, s_t1, v_ego_t1)
                if counter_value > 10:
                    print('!!! reward counter full up !!!')
                else:
                    agent.replay_memory(s_t, v_ego_t, action, reward, s_t1, v_ego_t1, done)
                agent.train_replay()

            s_t = s_t1
            v_ego_t = v_ego_t1
            v_ego = v_ego_1
            agent.t = agent.t + 1

            # # reward离散值计数，当小于0.01的reward连续出现10次，终止环境
            # if counter_value > 5:
            #     done = 1
            #     counter_value = 0
            #     print('!!! reward counter full up !!!')

            print("EPISODE",  e, "TIMESTEP", agent.t, "EXPLORE", noise.sigma, "/ ACTION", format(action, '.4f'), "/ REWARD", format(reward, '.4f'), "Avg REWARD:",
                    sum(rewardTot)/len(rewardTot) , "/ EPISODE LENGTH", episode_len, "/ time " , time_cost, a_ego1, 'loss_actor', agent.history_loss_actor, 'loss_critic', agent.history_loss_critic)
            format_str = 'EPISODE: %d TIMESTEP: %d EPISODE_LENGTH: %d ACTION: %.4f REWARD: %.4f Avg_REWARD: %.4f training_Loss: %.4f Q_MAX: %.4f gap: %.4f  v_ego: %.4f v_lead: %.4f time: %.0f a_ego: %.4f loss_actor: %.4f'
            text = (format_str % (e, agent.t, episode_len, action, reward, sum(rewardTot)/len(rewardTot), agent.history_loss_critic, agent.max_Q, gap, v_ego1, v_lead, time.time()-start_time, a_ego1, agent.history_loss_actor))
            print_out(train_log, text)
            if done:
                episodes.append(e)
                # Save model for every 2 episode
                if agent.train and (e % 5 == 0):
                    agent.save_model(os.path.join(PATH_, "save_model_{}.h5".format(e)))
                print("episode:", e, "  memory length:", len(agent.memory), " episode length:", episode_len)
                if done == 1:
                    reset()
                    time.sleep(0.5)
            print('Data receive from unity, time:', int(round(time.time() * 1000) - start_count_time))

        # 迭代过程测试
        # 测试
        if e % 5 == 0 and e >= 40:
            if done == 2:
                print("!!!TESTING!!! Continued epicode for testing!!!")
                done = 0
                UnityReset = 1
                episode_len = 0
            else:
                print('done value:', done)
                print("!!! STARTING TEST !!!")
                done = 1
                s_t, v_ego_t, v_ego, v_lead, remoteHost, remotePort = Recv_data_Format(unity_Block_size, done)
                done = 0
                UnityReset = 0
                episode_len = 0

            while done == 0:
                episode_len, action, time_cost, UnityReset = Send_data_Format(remoteHost, remotePort, s_t, v_ego_t,
                                                                              episode_len, UnityReset)
                reward, done, gap, v_ego1, v_lead, a_ego1, v_ego_1, s_t1, v_ego_t1 = Recv_data_Format(
                    unity_Block_size,
                    done, v_ego,
                    v_lead, action,
                    episode_len, s_t,
                    v_ego_t)

                s_t = s_t1
                v_ego_t = v_ego_t1
                v_ego = v_ego_1
                agent.t = agent.t + 1

                print("!!!TESTING!!!", "EPISODE", e, "TIMESTEP", agent.t, "/ ACTION", action, "/ REWARD",
                      format(reward, '.4f'), "/ EPISODE LENGTH", episode_len, "/ GAP ",
                      gap, "v_ego", v_ego.item(), "v_lead", v_lead, "/ a_ego ", a_ego1, 'loss_actor',
                      agent.history_loss_actor, 'loss_critic', agent.history_loss_critic)
                format_str = f'EPISODE: {e} TIMESTEP: {agent.t} EPISODE_LENGTH: {episode_len} ' \
                             f'ACTION: {action:.4f} REWARD: {reward:.4f} ' \
                             f'Avg_REWARD: {sum(rewardTot) / len(rewardTot):.4f} gap: {gap:.4f} ' \
                             f'v_ego: {v_ego.item():.4f} v_lead: {v_lead:.4f} a_ego: {a_ego1:.4f} ' \
                             f'loss_actor: {agent.history_loss_actor:.4f} loss_critic: {agent.history_loss_critic:.4f}'
                print_out(test_log, format_str)

                if done:
                    # Save model for every 2 episode
                    print("episode:", e, "  memory length:", len(agent.memory), " episode length:", episode_len)
                    if done == 1:
                        reset()
                        time.sleep(0.5)
        if e > 100:
            noise.sigma *= .95
        counter_value = 0
