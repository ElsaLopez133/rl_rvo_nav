import gym
import gym_env
from pathlib import Path
import pickle
import sys
from rl_rvo_nav.policy_test.post_train import post_train
import argparse
import os, pwd
import numpy as np
import yaml

parser = argparse.ArgumentParser(description='policy test')
parser.add_argument('--policy_type', default='drl')
parser.add_argument('--model_path', default='pre_trained')
parser.add_argument('--model_name', default='pre_train_check_point_1000.pt')
parser.add_argument('--arg_name', default='pre_train')
parser.add_argument('--world_name', default='test_world.yaml')  # test_world_lines.yaml; test_world_dyna_obs.yaml
parser.add_argument('--render', action='store_true')
parser.add_argument('--robot_number', type=int, default='10')
parser.add_argument('--num_episodes', type=int, default='30')
parser.add_argument('--dis_mode', type=int, default='3')
#parser.add_argument('--dis_mode', type=int, default='0') # to choose manually the goal
#parser.add_argument('--experiment', type=int, default='1')
#parser.add_argument('--trial', type=int, default='1')
#parser.add_argument('--vmax_linear', type=float, default='1.5')
#parser.add_argument('--vmax_angular', type=float, default='1.5')
#parser.add_argument('--step', type=float, default='0.1')
parser.add_argument('--save', action='store_true')
parser.add_argument('--full', action='store_true')
parser.add_argument('--show_traj', action='store_true')
parser.add_argument('--once', action='store_true')
parser.add_argument('--use_cpu', action='store_true')

policy_args = parser.parse_args()

cur_path = Path(__file__).parent
model_base_path = str(cur_path) + '/' + policy_args.model_path
args_path = model_base_path + '/' + policy_args.arg_name

# args from train
r = open(args_path, 'rb')
args = pickle.load(r)
#args.neighbors_region = 4
if policy_args.use_cpu:
    args.use_gpu = False

# Read YAML file. We want to run sveral times with different values of the raidus

with open(policy_args.world_name, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
    radius_list = data_loaded['robots']['radius_list']
    neighbors_region_list = data_loaded['robots']['neighbors_region_list']
    vmax_list = data_loaded['robots']['vmax_list']
    #vmax_linear = data_loaded['robots']['vel_max'][0]
    #vmax_angular = data_loaded['robots']['vel_max'][1]
    step = data_loaded['world']['step_time']
    print(data_loaded)

    for k in range(len(radius_list)):
        radius = radius_list[k]
        print('Running experiment for radius {0}'.format(radius))

        for i in range(len(neighbors_region_list)):

            if policy_args.policy_type == 'drl':
                # fname_model = save_path_string +'_check_point_250.pt'
                fname_model = model_base_path + '/' + policy_args.model_name
                policy_name = 'rl_rvo'
            if policy_args.world_name == 'test_world_dyna_obs.yaml':
                policy_name = policy_name + '_static_obstacle'

            args.neighbors_region = neighbors_region_list[i]
            print('Running experiment for nr {0}'.format(args.neighbors_region))

            exp_name = policy_name + '_' + str(policy_args.robot_number) + '_dis' + str(policy_args.dis_mode) + '_nr' +  str(args.neighbors_region) # + '_radius' + str(radius) #+ '_episodes' + str(policy_args.num_episodes) + '_radius' + str(radius).replace('.','-')
            policy_name = exp_name
            
            for j in range(len(vmax_list)):
                vmax_linear = vmax_list[j]
                vmax_angular = vmax_list[j]
                print('Running experiment for vmax_linear and angular {0}  {1}'.format(vmax_linear,vmax_angular))

                # we create/ check the folders where we are going to save the data
                os.makedirs(str(pwd.getpwuid(os.getuid()).pw_dir) + '/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/Experiments/'+ exp_name, exist_ok=True)
                os.makedirs(str(pwd.getpwuid(os.getuid()).pw_dir) + '/catkin_ws/src/kale_bot/external/rl_rvo_nav/rl_rvo_nav/Experiments/'+ exp_name + '/images', exist_ok=True)

                env = gym.make('mrnav-v1',abs_action_list = [],a_inc_list = [], world_name=policy_args.world_name, robot_number=policy_args.robot_number, neighbors_region=args.neighbors_region, neighbors_num=args.neighbors_num, robot_init_mode=policy_args.dis_mode, env_train=False, random_bear=args.random_bear, random_radius=args.random_radius, reward_parameter=args.reward_parameter, goal_threshold=0.2, full=policy_args.full)

                pt = post_train(env, num_episodes=policy_args.num_episodes, reset_mode=policy_args.dis_mode, render=policy_args.render, std_factor=0.00001, acceler_vel=1.0, max_ep_len=300, neighbors_region=args.neighbors_region, neighbor_num=args.neighbors_num, args=args, exp_name=exp_name, save=policy_args.save, show_traj=policy_args.show_traj, figure_format='eps')

                pt.policy_test(policy_args.policy_type, fname_model, policy_name, result_path=str(cur_path), result_name='/result.txt', figure_save_path= None , ani_save_path=cur_path / 'gif', policy_dict=True,  once=policy_args.once, vmax_angular = vmax_angular, vmax_linear = vmax_linear, step = step)
