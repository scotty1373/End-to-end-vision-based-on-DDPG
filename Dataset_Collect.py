import base64
import copy
import datetime as dt
import io
import os
import random
import socket
import sys
import threading
import time
from collections import deque
from net_builder import Data_dim_reduce as build_model
import numpy as np
import skimage
import torch
from torch.autograd import Variable

from PIL import Image
from skimage import color, exposure, transform
import cv2


EPISODES = 500
img_rows, img_cols = 80, 80
# Convert image into gray scale
# We stack 8 frames, 0.06*8 sec
img_channels = 4 
unity_Block_size = 65536
# PATH_MODEL = 'C:/dl_data/Python_Project/save_model/'
# PATH_LOG = 'C:/dl_data/Python_Project/train_log/'
PATH_MODEL = 'save_Model'
PATH_LOG = 'train_Log'
PATH_DATASET = 'dataSet_image'
time_Feature = round(time.time())
random_index = np.random.permutation(img_channels)

Data_Collect = True


class DQNAgent:
    def __init__(self, state_size, action_size, device_):
        self.t = 0
        self.max_Q = 0
        self.trainingLoss = 0
        self.train = True

        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.device = device_

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if self.train:
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 0
            self.initial_epsilon = 0
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 100
        self.explore = 4000

        # Create replay memory using deque
        self.memory = deque(maxlen=32000)

        # Create main model and target model
        self.model = build_model().to(self.device)
        self.target_model = build_model().to(self.device)

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss = torch.nn.MSELoss()

    def process_image(self, obs):
        obs = skimage.color.rgb2gray(obs)
        return obs
        # camera_info = CamInfo({
        #     "f_x": 500/5*8,         # focal length x
        #     "f_y": 500/5*8,         # focal length y
        #     "u_x": 200,             # optical center x
        #     "u_y": 200,             # optical center y
        #     "camera_height": 1400,  # camera height in `mm`
        #     "pitch": 90,            # rotation degree around x
        #     "yaw": 0                # rotation degree around y
        # })
        # ipm_info = CamInfo({
        #     "input_width": 400,
        #     "input_height": 400,
        #     "out_width": 80,
        #     "out_height": 80,
        #     "left": 0,
        #     "right": 400,
        #     "top": 200,
        #     "bottom": 400
        # })
        # ipm_img = IPM(camera_info, ipm_info)
        # out_img = ipm_img(obs)
        # if gap < 10:
        #     skimage.io.imsave('outimage_' + str(gap) + '.png', out_img)

        # return out_img

    def update_target_model(self):
        # 解决state_dict浅拷贝问题
        weight_model = copy.deepcopy(self.model.state_dict())
        self.target_model.load_state_dict(weight_model)

    # Get action from model using epsilon-greedy policy
    def get_action(self, Input):
        if np.random.rand() <= self.epsilon:
            # print("Return Random Value")
            # return random.randrange(self.action_size)
            return np.random.uniform(-1, 1)
        else:
            # print("Return Max Q Prediction")
            q_value = self.model(Input[0], Input[1])
            # Convert q array to steering value
            return linear_unbin(q_value[0])

    def replay_memory(self, state, v_ego, action, reward, next_state, nextV_ego, done):
        self.memory.append((state, v_ego, action, reward, next_state, nextV_ego, done, self.t))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

    # @profile
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        '''
        torch.float64对应torch.DoubleTensor
        torch.float32对应torch.FloatTensor
        '''
        state_t, v_ego_t, action_t, reward_t, state_t1, v_ego_t1, terminal, step = zip(*minibatch)
        state_t = Variable(torch.Tensor(state_t).squeeze().to(self.device))
        state_t1 = Variable(torch.Tensor(state_t1).squeeze().to(self.device))
        v_ego_t = Variable(torch.Tensor(v_ego_t).squeeze().to(self.device))
        v_ego_t1 = Variable(torch.Tensor(v_ego_t1).squeeze().to(device))

        self.optimizer.zero_grad()

        targets = self.model(state_t, v_ego_t)
        self.max_Q = torch.max(targets[0]).item()
        target_val = self.model(state_t1, v_ego_t1)
        target_val_ = self.target_model(state_t1, v_ego_t1)
        for i in range(batch_size):
            if terminal[i] == 1:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = torch.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])
        logits = self.model(state_t, v_ego_t)
        loss = self.loss(logits, targets)
        loss.backward()
        self.optimizer.step()
        self.trainingLoss = loss.item()

    def load_model(self, name):
        checkpoints = torch.load(name)
        self.model.load_state_dict(checkpoints['model'])
        self.optimizer.load_state_dict(checkpoints['optimizer'])

    # Save the model which is under training
    def save_model(self, name):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, name)


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


def linear_bin(a):
    """
    Convert a value to a categorical array.
    Parameters
    ----------
    a : int or float
        A value between -1 and 1
    Returns
    -------
    list of int
        A list of length 21 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 20))
    arr = np.zeros(21)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    """
    Convert a categorical array to value.
    See Also
    --------
    linear_bin
    """
    arr = arr.data.cpu().numpy()
    if not len(arr) == 21:
        raise ValueError('Illegal array length, must be 21')
    b = np.argmax(arr)
    a = b * 2 / 20 - 1
    return a
# def oberve():
#   revcData, (remoteHost, remotePort) = sock.recvfrom(65536)


def decode(revcData, v_ego = 0, force = 0, episode_len = 0):
    # received data processing
    revcList = str(revcData).split(',', 4)
    gap = revcList[0][2:]                    # distance between vehicles
    v_ego1 = revcList[1]                      # speed of egoVehicle
    v_lead = revcList[2]                     # speed of leadVehicle
    a_ego1 = revcList[3]                      # acceleration of egoVehicle
    img = base64.b64decode(revcList[4])      # image from mainCamera
    image = Image.open(io.BytesIO(img))
    image.save(f'./{PATH_DATASET}/data_{time_Feature}/{agent.t:05}.jpg', quality=95)
    # image resize, 双线性插值
    image = image.resize((80, 80), resample=Image.BILINEAR)
    image = np.array(image)
    done = 0
    reward = CalReward(float(gap), float(v_ego), float(v_lead), force)
    if float(gap) <= 3 or float(gap) >= 300:
        done = 1  
        reward = -1.0
    elif episode_len > 480:
        done = 2 
        # reward = CalReward(float(gap), float(v_ego), float(v_lead), force)

    return image, reward, done, float(gap), float(v_ego1), float(v_lead), float(a_ego1)


def CalReward(gap, v_ego, v_lead, force):
    Rd, Rp = 0, 0
    Rd, Rp = 0, 1

    if force>0:
        a = 3.5*force
    else:
        a = 5.5*force
        
    L0 = -3.037
    L1 = -0.591
    L3 = -1.047e-3
    L4 = -1.403
    L5 = 2.831e-2
    L8 = -7.98e-2
    L11 = 3.535e-3
    L12 = -0.243
    Rp = (L0 + L1*v_ego + L3*(v_ego**3) + L4* v_ego * a + L5*(v_ego**2) + L8 * (v_ego**2) * a + L11 * (v_ego**3) * a + L12 * v_ego * (a**2))
    if Rp>0:
        Rp = 0
    Rp = Rp + 195
    if v_ego > 40:
        Rp = 0
    # reward for gap
    if gap >= 40 and gap <= 60:
        Rd = 1
    elif gap >= 30 and gap < 40:
        Rd = 0.5
    elif gap > 60 and gap <= 100:
        Rd = 0.5        
    else:
        Rd = 0.0
    # if gap >= 40 and gap <= 60:
    #     Rd = 1
    # elif gap >= 30 and gap < 40:
    #     Rd = np.power(1.29, (gap - 40))
    # elif gap > 60 and gap <= 100:
    #     Rd = np.power(1.29, (-gap + 60))     
    # else:
    #     Rd = 0

    # return Rp*Rd
    return Rp*Rd/195.0


def reset():
    strr = str(3) + ',' + '0.0'
    sendDataLen = sock.sendto(strr.encode(), (remoteHost, remotePort))


def print_out(file, text):
    file.write(text + '\n')
    file.flush()
    sys.stdout.flush()


# @profile
def thread_Train_init():
    global agent
    step_epsode = 0
    while True:
        if len(agent.memory) < agent.train_start:
            time.sleep(5)
            continue
        agent.train_replay()
        time.sleep(0.1)
        step_epsode += 1
        # print('train complete in num: %s' %str(step_epsode))


def log_File_path(path):
    # date = str(dt.date.today()).split('-')
    # date_concat = date[1] + date[2]
    date_concat = time_Feature
    train_log = open(os.path.join(path, 'train_log_{}.txt'.format(date_concat)), 'w')
    del date_concat
    return train_log


def random_sample(state_t, v_t, state_t1, v_t1):
    # random_index = np.random.permutation(img_channels)
    state_t = state_t[:, :, :, random_index]
    v_t = v_t[:, random_index]
    state_t1 = state_t1[:, :, :, random_index]
    v_t1 = v_t1[:, random_index]
    return state_t, v_t, state_t1, v_t1


def Recv_data_Format(byte_size, _done, v_ego=None, action=None, episode_len=None, s_t=None, v_ego_t=None):
    if _done != 0: 
        revcData, (remoteHost, remotePort) = sock.recvfrom(byte_size)
        image, _, _, gap, v_ego, _, a_ego = decode(revcData)
        x_t = agent.process_image(image)

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
        v_ego_t = np.array((v_ego, v_ego, v_ego, v_ego))

        # In Keras, need to reshape
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) #1*80*80*4
        v_ego_t = v_ego_t.reshape(1, v_ego_t.shape[0]) #1*4
        return s_t, v_ego_t, v_ego, remoteHost, remotePort
    else:
        revcData, (remoteHost, remotePort) = sock.recvfrom(byte_size)
        image, reward, done, gap, v_ego1, v_lead, a_ego1 = decode(revcData, v_ego, action, episode_len)

        x_t1 = agent.process_image(image)
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])     # 1x1x80x80
        s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)    # 1x4x80x80
        v_ego_1 = np.array(v_ego1)
        v_ego_1 = np.expand_dims(v_ego_1, -1)
        v_ego_1 = np.expand_dims(v_ego_1, -1)
        v_ego_t1 = np.append(v_ego_1, v_ego_t[:, :3], axis=1)   # 1x4
        return reward, done, gap, v_ego1, v_lead, a_ego1, v_ego_1, s_t1, v_ego_t1


# def Send_data_Format(remoteHost, remotePort, onlyresetloc, s_t, v_ego_t):
def Send_data_Format(remoteHost, remotePort, s_t, v_ego_t, episode_len, UnityReset):
    pred_time_pre = dt.datetime.now()
    episode_len = episode_len + 1            
    # Get action for the current state and go one step in environment
    s_t = torch.Tensor(s_t).to(device)
    v_ego_t = torch.Tensor(v_ego_t).to(device)
    force = agent.get_action([s_t, v_ego_t])
    action = force
      
    if UnityReset == 1: 
        strr = str(4) + ',' + str(action)
        UnityReset = 0
    else:
        strr = str(1) + ',' + str(action)
    
    sendDataLen = sock.sendto(strr.encode(), (remoteHost, remotePort))      # 0.06s later receive
    pred_time_end = dt.datetime.now()
    time_cost = pred_time_end - pred_time_pre
    return episode_len, action, time_cost, UnityReset


def Model_save_Dir(PATH, time):
    path_to_return = os.path.join(PATH, 'save_model_{}'.format(time)) + '/'
    if not os.path.exists(path_to_return):
        os.mkdir(path_to_return)   
    return path_to_return
   
    
if __name__ == "__main__":
    if not os.path.exists('./' + PATH_LOG):
        os.mkdir(os.path.join(os.getcwd().replace('\\', '/'), PATH_LOG))
    if not os.path.exists('./' + PATH_MODEL):
        os.mkdir(os.path.join(os.getcwd().replace('\\', '/'), PATH_MODEL))

    os.mkdir(os.path.join(os.getcwd(), PATH_DATASET, f'data_{time_Feature}').replace('\\', '/'))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 8001))

    device = torch.device('cpu')

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)
    action_size = 21    # env.action_space.n # Steering and Throttle

    train_log = log_File_path(PATH_LOG)
    PATH_ = Model_save_Dir(PATH_MODEL, time_Feature)
    agent = DQNAgent(state_size, action_size, device)
    episodes = []

    if not agent.train:
        print("Now we load the saved model")
        agent.load_model("C:/DRL_data/Python_Project/Enhence_Learning/save_Model/save_model_1627300305/save_model_248.pt")
    else:
        # train_thread = threading.Thread(target=thread_Train_init)
        # train_thread.start()
        print('Thread Ready!!!')
    done = 0

    for e in range(EPISODES):      
        print("Episode: ", e)
        # Multi Thread
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
            s_t, v_ego_t, v_ego, remoteHost, remotePort = Recv_data_Format(unity_Block_size, done)
            done = 0
            UnityReset = 0
            episode_len = 0

        while done == 0:
            start_time = time.time()
            if agent.t % 1000 == 0:
                rewardTot = []
            episode_len, action, time_cost, UnityReset = Send_data_Format(remoteHost, remotePort, s_t, v_ego_t, episode_len, UnityReset)
            reward, done, gap, v_ego1, v_lead, a_ego1, v_ego_1, s_t1, v_ego_t1 = Recv_data_Format(unity_Block_size, done, v_ego, action, episode_len, s_t, v_ego_t)
            rewardTot.append(reward)
            start_count_time = int(round(time.time() * 1000))
            
            if agent.train:
                # s_t, v_ego_t, s_t1, v_ego_t1 = random_sample(s_t, v_ego_t, s_t1, v_ego_t1)
                agent.replay_memory(s_t, v_ego_t, np.argmax(linear_bin(action)), reward, s_t1, v_ego_t1, done)
                agent.train_replay()

            s_t = s_t1
            v_ego_t = v_ego_t1
            v_ego = v_ego_1
            agent.t = agent.t + 1

            print("EPISODE",  e, "TIMESTEP", agent.t,"/ ACTION", action, "/ REWARD", reward, "Avg REWARD:",
                    sum(rewardTot)/len(rewardTot) , "/ EPISODE LENGTH", episode_len, "/ Q_MAX " ,
                    agent.max_Q, "/ time " , time_cost, a_ego1)
            format_str = ('EPISODE: %d TIMESTEP: %d EPISODE_LENGTH: %d ACTION: %.4f REWARD: %.4f Avg_REWARD: %.4f training_Loss: %.4f Q_MAX: %.4f gap: %.4f  v_ego: %.4f v_lead: %.4f time: %.0f a_ego: %.4f')
            text = (format_str % (e, agent.t, episode_len, action, reward, sum(rewardTot)/len(rewardTot), agent.trainingLoss*1e3, agent.max_Q, gap, v_ego1, v_lead, time.time()-start_time, a_ego1))
            print_out(train_log, text)
            if done:
                agent.update_target_model()
                episodes.append(e)
                # Save model for every 2 episode
                if agent.train and (e % 2 == 0):
                    agent.save_model(os.path.join(PATH_, "save_model_{}.h5".format(e)))
                print("episode:", e, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon, " episode length:", episode_len)
                if done == 1: 
                    reset()
                    time.sleep(0.5)
            print('Data receive from unity, time:', int(round(time.time() * 1000) - start_count_time))

        # Tensorboard_saver = tf.summary.FileWriter('E:/Python_Project/Enhence_Learning/Tensorboard/', tf.get_default_graph())
        # lp = LineProfiler()
        # lp_wrapper = lp(agent.train_replay())
        # lp.print_stats()
