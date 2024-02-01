import math
import torch
import torch.optim as optim
from tankbattle.env.utils import Utils
from tankbattle.env.engine import TankBattle
from model.net import DeepQNetwork
from torch.autograd import Variable
from model.replay_buffer import PrioritizedBuffer
from model.train_information import TrainInformation


device = "cuda" if torch.cuda.is_available() else "cpu"
data_save_path = "output/train/"
param_save_path = "parameter/"
num_episodes = 2000
batch_size = 64
initial_learning = 0
target_update_frequency = 100
buffer_capacity = 2048
learning_rate = 1e-4
gamma = 1.00
beta_start = 0.4
total_frames = 10000
eps_f = 0.01
eps_s = 0.4
decay_rate = 100000

def td_loss(net_model, target_network, replay_buff, gamma_val, device_val, batch_size_val, beta_val_param):
    s, a, r, s_next, done, indices, weights = replay_buff.sample(batch_size_val, beta_val_param)
    s = Variable(torch.FloatTensor(s)).to(device_val)
    s_next = Variable(torch.FloatTensor(s_next)).to(device_val)
    a = Variable(torch.LongTensor(a)).to(device_val)
    r = Variable(torch.FloatTensor(r)).to(device_val)
    done = Variable(torch.FloatTensor(done)).to(device_val)
    weights = Variable(torch.FloatTensor(weights)).to(device_val)
    
    q_values = net_model(s)
    next_q_values = target_network(s_next)
    
    q_value = q_values.gather(1, a.unsqueeze(-1)).squeeze(-1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = r + gamma_val * next_q_value * (1 - done)
    
    loss = (q_value - expected_q_value.detach()).pow(2)*weights
    prios = loss + 1e-5
    loss = loss.mean()
    
    loss.backward()
    replay_buff.update_priorities(indices, prios.data.cpu().numpy())

def update_beta(episode_num):
    beta_val_new = beta_start + episode_num * (1.0 - beta_start) / total_frames
    return min(1.0, beta_val_new)

def update_epsilon(episode):
    epsilon = eps_f + (eps_s - eps_f) * math.exp(-1 * ((episode + 1) / decay_rate))
    return epsilon

def train(game, net_model, target_network, optimizer_val, replay_buff, device_val):
    info = TrainInformation()
    rewards = []
    max_reward = -500
    best_episode = 0
    episode_list = []

    for episode in range(num_episodes):
        state = Utils.resize_image(game.reset())
        episode_reward = 0

        for _ in range(20000):
            epsilon = update_epsilon(info.index)
            beta_val = update_beta(info.index) if len(replay_buff) > batch_size else 0.4

            action = net_model.select_action(state, epsilon, device_val)
            next_state, reward, done, _ = game.step(action)
            next_state = Utils.resize_image(next_state)
            reward = reward[0]
            episode_reward += reward

            if reward >= -1000:
                replay_buff.push(state, action, reward, next_state, done)

            if len(replay_buff) > initial_learning and not info.index % target_update_frequency:
                target_network.load_state_dict(net_model.state_dict())
            optimizer_val.zero_grad()
            td_loss(net_model, target_network, replay_buff, gamma, device_val, batch_size, beta_val)
            optimizer_val.step()

            if done:
                break

            state = next_state

        rewards.append(episode_reward)
        print(f"Episode: {episode}, score: {episode_reward}, epsilon: {epsilon}, max_score: {max_reward}, and best_episode: {best_episode}")
        episode_list.append(episode)

        if episode_reward >= max_reward:
            max_reward = episode_reward
            best_episode = episode
            Utils.save_model(net_model, param_save_path, episode, epsilon, episode_reward)

if __name__ == "__main__":
    game_env = TankBattle(render=True, player1_human_control=False, player2_human_control=False,
                          two_players=False, speed=60, debug=False, frame_skip=5)
    init_state = game_env.reset()
    state = Utils.resize_image(init_state)
    network_model = DeepQNetwork(obs_shape=state.shape, action_count=game_env.get_num_of_actions())
    target_network = DeepQNetwork(obs_shape=state.shape, action_count=game_env.get_num_of_actions())
    replay_buff = PrioritizedBuffer(buffer_capacity)
    optimizer = optim.Adam(network_model.parameters(), lr=learning_rate)
    train(game_env, network_model, target_network, optimizer, replay_buff, device)
