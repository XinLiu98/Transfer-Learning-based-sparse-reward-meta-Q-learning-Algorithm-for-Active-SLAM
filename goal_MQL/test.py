import argparse

import pandas as pd
import torch
import os
import time
import sys
import numpy as np
from collections import deque
import random

import matplotlib.pyplot as plt#生成图要用matplotlib模块，若没有则得先安装。

import rlkit.envs.agent
from misc.utils import create_dir, dump_to_json, CSVWriter 
from misc.torch_utility import get_state
#form rlkit.envs.agent import Agent
from misc.utils import set_global_seeds, safemean, read_json
from misc import logger
from algs.MQL.buffer import Buffer
from algs.MQL.mql import MQL
from tensorboardX import SummaryWriter
# from rlkit.envs.env_sample import EnvSampler
from rlkit.envs.environment_one_agent import GazeboDirEnv

import os


parser = argparse.ArgumentParser()

# Optim params
parser.add_argument('--lr', type=float, default=0.0003, help = 'Learning rate')
parser.add_argument('--replay_size', type=int, default = 1e6, help ='Replay buffer size int(1e6)')
parser.add_argument('--ptau', type=float, default=0.005 , help = 'Interpolation factor in polyak averaging')
parser.add_argument('--gamma', type=float, default=0.99, help = 'Discount factor [0,1]')
parser.add_argument("--burn_in", default=1e4, type=int, help = 'How many time steps purely random policy is run for') 
parser.add_argument("--total_timesteps", default=5e6, type=float, help = 'Total number of timesteps to train on')
parser.add_argument("--expl_noise", default=0.2, type=float, help='Std of Gaussian exploration noise')
parser.add_argument("--batch_size", default=256, type=int, help = 'Batch size for both actor and critic')
parser.add_argument("--policy_noise", default=0.3, type=float, help =' Noise added to target policy during critic update')
parser.add_argument("--noise_clip", default=0.5, type=float, help='Range to clip target policy noise')
parser.add_argument("--policy_freq", default=2, type=int, help='Frequency of delayed policy updates')
parser.add_argument('--hidden_sizes', nargs='+', type=int, default = [300, 300], help = 'indicates hidden size actor/critic')

# General params
parser.add_argument('--env_name', type=str, default='ant-goal')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--alg_name', type=str, default='mql')

parser.add_argument('--disable_cuda', default=False, action='store_true')
parser.add_argument('--cuda_deterministic', default=False, action='store_true')
parser.add_argument("--gpu_id", default=0, type=int)

parser.add_argument('--log_id', default='empty')
parser.add_argument('--check_point_dir', default='./ck')
parser.add_argument('--log_dir', default='./log_dir')
parser.add_argument('--log_interval', type=int, default=1, help='log interval, one log per n updates') # todo 10---1
parser.add_argument('--save_freq', type=int, default = 50)#todo 250--50
parser.add_argument("--eval_freq", default=5e3, type=float, help = 'How often (time steps) we evaluate')    

# Env
parser.add_argument('--env_configs', default='./configs/pearl_envs.json')
parser.add_argument('--max_path_length', type=int, default = 600) #todo 5.3 1200---2000
parser.add_argument('--enable_train_eval', default=False, action='store_true') #TODO 4.25 F--T
parser.add_argument('--enable_promp_envs', default=False, action='store_true')  #TODO 4.25 F--T
parser.add_argument('--num_initial_steps',  type=int, default = 1000)
parser.add_argument('--unbounded_eval_hist', default=False, action='store_true')

#context
parser.add_argument('--hiddens_conext', nargs='+', type=int, default = [30], help = 'indicates hidden size of context next')
parser.add_argument('--enable_context', default=True, action='store_true')
parser.add_argument('--only_concat_context', type=int, default = 3, help =' use conext')
parser.add_argument('--num_tasks_sample', type=int, default = 5)
parser.add_argument('--num_train_steps', type=int, default = 500)
parser.add_argument('--min_buffer_size', type=int, default = 100000, help = 'this indicates a condition to start using num_train_steps')
parser.add_argument('--history_length', type=int, default = 30)

#other params
parser.add_argument('--beta_clip', default=1.0, type=float, help='Range to clip beta term in CSC')
parser.add_argument('--snapshot_size', type=int, default = 2000, help ='Snapshot size for a task')
parser.add_argument('--prox_coef', default=0.1, type=float, help ='Prox lambda')
parser.add_argument('--meta_batch_size', default=10, type=int, help ='Meta batch size: number of sampled tasks per itr')
parser.add_argument('--enable_adaptation', default=True, action='store_true')
parser.add_argument('--main_snap_iter_nums', default=100, type=int, help ='how many times adapt using train task but with csc')
parser.add_argument('--snap_iter_nums', default=10, type=int, help ='how many times adapt using eval task')
parser.add_argument('--type_of_training', default='td3', help = 'td3')
parser.add_argument('--lam_csc', default=0.50, type=float, help='logisitc regression reg, smaller means stronger reg')
parser.add_argument('--use_ess_clipping', default=False, action='store_true')
parser.add_argument('--enable_beta_obs_cxt', default=False, action='store_true', help='if true concat obs + context')
parser.add_argument('--sampling_style', default='replay', help = 'replay')
parser.add_argument('--sample_mult',  type=int, default = 5, help ='sample multipler of main_iter for adapt method')
parser.add_argument('--use_epi_len_steps', default=True, action='store_true')
parser.add_argument('--use_normalized_beta', default=False, action='store_true', help = 'normalized beta_score')
parser.add_argument('--reset_optims', default=False, action='store_true', help = 'init optimizers at the start of adaptation')
parser.add_argument('--lr_milestone', default = -1, type=int, help = 'reduce learning rate after this epoch')
parser.add_argument('--lr_gamma', default = 0.8, type=float, help = 'learning rate decay')
#parser.add_argument('--MAX_STEPS_TRAINING',  type=int, default =400)
#parser.add_argument('--NUM_TRAIN_REPEAT',  type=int, default =1)
#parser.add_argument('--start_timesteps',  type=int, default =1000)


def update_lr(eparams, iter_num, alg_mth):
    #######
    # initial_lr if i < reduce_lr
    # otherwise initial_lr * lr_gamma
    #######
    if iter_num > eparams.lr_milestone:
        new_lr = eparams.lr * eparams.lr_gamma

        for param_group in alg_mth.actor_optimizer.param_groups:
            param_group['lr'] = new_lr

        for param_group in alg_mth.critic_optimizer.param_groups:
            param_group['lr'] = new_lr
        print("---------")
        print("Actor (updated_lr):\n ",  alg_mth.actor_optimizer)
        print("Critic (updated_lr):\n ", alg_mth.critic_optimizer)
        print("---------")

def take_snapshot(args, ck_fname_part, model, update):
    '''
        This fucntion just save the current model and save some other info
    '''
    fname_ck =  ck_fname_part + '.pt'
    fname_json =  ck_fname_part + '.json'
    curr_state_actor = get_state(model.actor)
    curr_state_critic = get_state(model.critic)
    # curr_state_advantage_generator_net = get_state(model.advantage_generator_net)
    # curr_state_actor_optimizer = get_state(model.actor_optimizer)
    # curr_state_critic_optimizer = get_state(model.critic_optimizer)

    print('Saving a checkpoint for iteration %d in %s' % (update, fname_ck))
    checkpoint = {
                    'args': args.__dict__,
                    'model_states_actor': curr_state_actor,
                    'model_states_critic': curr_state_critic,
                    # 'model_states_advantage_generator_net': curr_state_advantage_generator_net,
                    # 'model_state_actor_optimizer': curr_state_actor_optimizer,
                    # 'model_state_critic_optimizer': curr_state_critic_optimizer,
                 }
    torch.save(checkpoint, fname_ck)

    del checkpoint['model_states_actor']
    del checkpoint['model_states_critic']
    # del checkpoint['model_states_advantage_generator_net']
    # del checkpoint['model_state_actor_optimizer']
    # del checkpoint['model_state_critic_optimizer']
    del curr_state_actor
    del curr_state_critic
    # del curr_state_advantage_generator_net
    # del curr_state_actor_optimizer
    # del curr_state_critic_optimizer
    
    dump_to_json(fname_json, checkpoint)

def setup_logAndCheckpoints(args):

    # create folder if not there
    create_dir(args.check_point_dir)

    fname = str.lower(args.env_name) + '_' + args.alg_name + '_' + args.log_id
    fname_log = os.path.join(args.log_dir, fname)
    fname_eval = os.path.join(fname_log, 'eval.csv')
    fname_adapt = os.path.join(fname_log, 'adapt.csv')

    return os.path.join(args.check_point_dir, fname), fname_log, fname_eval, fname_adapt

def make_env(eparams):
    '''
        This function builds env
    '''
    # since env contains np/sample function, need to set random seed here
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    ################
    # this is based on PEARL paper that fixes set of sampels
    ################
    from misc.env_meta import build_PEARL_envs
    env = build_PEARL_envs(
                           seed = eparams.seed,
                           env_name = eparams.env_name,
                           params = eparams,
                           )

    return env

def sample_env_tasks(env, eparams):
    '''
        Sample env tasks
    '''
    if eparams.enable_promp_envs == True:
        # task list created as [ train_task,..., train_task ,eval_task,..., eval_task]

        env_sampler = EnvSampler(env, max_path_length=eparams.max_path_length,
                                 start_timesteps=eparams.num_initial_steps)
        train_tasks = env_sampler.sample(eparams.n_train_tasks)
        eval_tasks = env_sampler.sample(eparams.n_eval_tasks)
        #train_tasks = env.sample_tasks(eparams.n_train_tasks)
       # eval_tasks  = env.sample_tasks(eparams.n_eval_tasks)


    else:
        # task list created as [ train_task,..., train_task ,eval_task,..., eval_task]
        tasks = env.get_all_task_idx()
        train_tasks = list(tasks[:eparams.n_train_tasks])
        eval_tasks = list(tasks[-eparams.n_eval_tasks:])

    return train_tasks, eval_tasks

def config_tasks_envs(eparams):
    '''
        Configure tasks parameters.
        Envs params and task parameters based on pearl paper:
        args like followings will be added:
        n_train_tasks   2
        n_eval_tasks    2
        n_tasks 2
        randomize_tasks true
        low_gear    false
        forward_backward    true
        num_evals   4
        num_steps_per_task  400
        num_steps_per_eval  400
        num_train_steps_per_itr 4000
    '''
    configs = read_json(eparams.env_configs)[eparams.env_name]
    temp_params = vars(eparams)
    for k, v in configs.items():
            temp_params[k] = v

def evaluate_policy(eval_env,
                    policy,
                    eps_num,
                    itr,
                    eparams,
                    msg ='Evaluation'):
    '''
        runs policy for X episodes and returns average reward
    '''
    if eparams.unbounded_eval_hist == True: # increase seq length to max_path_length
        eval_hist_len = eparams.max_path_length
        print('Eval uses unbounded_eval_hist of length: ', eval_hist_len)

    else:
        eval_hist_len = eparams.history_length
        print('Eval uses history of length: ', eval_hist_len)

    if eparams.enable_promp_envs == True:
        etasks  = eval_env.sample_tasks(eparams.n_eval_tasks)

    all_task_rewards = []
    dc_rewards = []
    avg_reward = 0
    avg_linear_speed = 0
    avg_angular_speed = 0
    #for _ in range(eparams.num_evals):

    ls = [[[] for i in range(itr)] for i in range(eparams.max_path_length)]
    ans = [[[] for i in range(itr)] for i in range(eparams.max_path_length)]
    rs = [[[] for i in range(itr)] for i in range(eparams.max_path_length)]
    for i in range(itr):
        obs = eval_env.reset()

        done = False
        arrive = False
        step = 0

        ### history ####
        rewards_hist = deque(maxlen=eval_hist_len)
        actions_hist = deque(maxlen=eval_hist_len)
        obsvs_hist   = deque(maxlen=eval_hist_len)

        rewards_hist.append(0)
        obsvs_hist.append(obs.copy())

        rand_action = np.random.normal(0, eparams.expl_noise, size=2)
        rand_action = rand_action.clip(-1, 1)
        actions_hist.append(rand_action.copy())


        while not (done or arrive) and step < eparams.max_path_length :

            np_pre_actions = np.asarray(actions_hist, dtype=np.float32).flatten() #(hist, action_dim) => (hist *action_dim,)
            np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32) #(hist, )
            np_pre_obsvs  = np.asarray(obsvs_hist, dtype=np.float32).flatten() #(hist, action_dim) => (hist *action_dim,)
            action = policy.select_action(np.array(obs), np.array(np_pre_actions), np.array(np_pre_rewards), np.array(np_pre_obsvs))
            print('evaluate_action',action)
            new_obs, reward, done, arrive, infos, linear_speed, angular_speed= eval_env.step(action)

            ls[step][i] = linear_speed
            ans[step][i] = angular_speed
            rs[step][i] = reward
            step += 1

            # new becomes old
            rewards_hist.append(reward)
            actions_hist.append(action.copy())
            obsvs_hist.append(obs.copy())
            obs = new_obs.copy()
        i += 1
    data1 = pd.DataFrame(ls)
    data1.to_csv('1.csv')
    data2 = pd.DataFrame(ans)
    data2.to_csv('2.csv')
    data3 = pd.DataFrame(rs)
    data3.to_csv('3.csv')

    return rs, ls, ans

if __name__ == "__main__":
    print(os.getcwd())
    args = parser.parse_args()
    print('------------')
    print(args.__dict__)
    print('------------')

    print('Read Tasks/Env config params and Update args')
    config_tasks_envs(args)
    print(args.__dict__)

    # if use mujoco-v2, then xml file should be ignore
    if ('-v2' in args.env_name):
        print('**** XML file is ignored since it is-v 2 ****')

    ##############################
    #### Generic setups
    ##############################
    CUDA_AVAL = torch.cuda.is_available()

    if not args.disable_cuda and CUDA_AVAL: 
        gpu_id = "cuda:" + str(args.gpu_id)
        device = torch.device(gpu_id)
        print("**** Yayy we use GPU %s ****" % gpu_id)

    else:                                                   
        device = torch.device('cpu')
        print("**** No GPU detected or GPU usage is disabled, sorry! ****")

    ####
    # train and evalution checkpoints, log folders, ck file names
    create_dir(args.log_dir, cleanup = True)
    # create folder for save checkpoints
    ck_fname_part, log_file_dir, fname_csv_eval, fname_adapt = setup_logAndCheckpoints(args)
    logger.configure(dir = log_file_dir)
    wrt_csv_eval = None


    #log_step_interval = 100  # 记录的步数间隔
    save_time = 0
    ##############################
    #### Init env, model, alg, batch generator etc
    #### Step 1: build env
    #### Step 2: Build model
    #### Step 3: Initiate Alg e.g. a2c
    #### Step 4: Initiate batch/rollout generator  
    ##############################

    ##### env setup #####
    env = make_env(args)
    #env = GazeboDirEnv()

#    env.render()

    ######### SEED ##########
    #  build_env already calls set seed,
    # Set seed the RNG for all devices (both CPU and CUDA)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.disable_cuda and CUDA_AVAL and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("****** cudnn.deterministic is set ******")

    ######### Build Networks
    max_action = 1.0
    action_1 = [0.0, 0.0]

    import models.networks as net

        ######
        # This part to add context network
        ######
    if args.enable_context == True:
        reward_dim = 1
        input_dim_context = 2+ reward_dim
        args.output_dim_conext =  (2+ reward_dim) * 2

        if args.only_concat_context == 3: # means use LSTM with action_reward_state as an input
             input_dim_context = 2 + reward_dim + 371
             actor_idim = [371 + args.hiddens_conext[0]]
             args.output_dim_conext = args.hiddens_conext[0]
             dim_others = args.hiddens_conext[0]

        else:
             raise ValueError(" %d args.only_concat_context is not supported" % (args.only_concat_context))

    actor_net = net.Actor(#action_space = env.action_space,
                              hidden_sizes =args.hidden_sizes,
                              input_dim = actor_idim,
                              max_action = max_action,
                              enable_context = args.enable_context,
                              hiddens_dim_conext = args.hiddens_conext,
                              input_dim_context = input_dim_context,
                              output_conext = args.output_dim_conext,
                              only_concat_context = args.only_concat_context,
                              history_length = args.history_length,
                              obsr_dim=371,
                              device = device
                              ).to(device)

    actor_target_net = net.Actor(#action_space = env.action_space,
                                    hidden_sizes =args.hidden_sizes,
                                    input_dim = actor_idim,
                                    max_action = max_action,
                                    enable_context = args.enable_context,
                                    hiddens_dim_conext = args.hiddens_conext,
                                    input_dim_context = input_dim_context,
                                    output_conext = args.output_dim_conext,
                                    only_concat_context = args.only_concat_context,
                                    history_length = args.history_length,
                                    obsr_dim=371,
                                    #obsr_dim=3630,
                                   # obsr_dim = env.observation_space.shape[0],
                                    device = device
                                     ).to(device)

    critic_net = net.Critic(#action_space = env.action_space,
                                hidden_sizes =args.hidden_sizes,
                                # input_dim=3634,
                                input_dim = (1,371),
                                #input_dim = (10,363),
                                enable_context = args.enable_context,
                                dim_others = dim_others,
                                hiddens_dim_conext = args.hiddens_conext,
                                input_dim_context = input_dim_context,
                                output_conext = args.output_dim_conext,
                                only_concat_context = args.only_concat_context,
                                history_length = args.history_length,
                                obsr_dim=371,
                                #obsr_dim=3630,
                                #obsr_dim = env.observation_space.shape[0],
                                device = device
                                ).to(device)

    critic_target_net = net.Critic(#action_space = env.action_space,
                                        hidden_sizes =args.hidden_sizes,
                                        #input_dim = env.observation_space.shape,
                                        input_dim=(1,371),
                                        #input_dim = (10,363),
                                        enable_context = args.enable_context,
                                        dim_others = dim_others,
                                        hiddens_dim_conext = args.hiddens_conext,
                                        input_dim_context = input_dim_context,
                                        output_conext = args.output_dim_conext,
                                        only_concat_context = args.only_concat_context,
                                        history_length = args.history_length,
                                        obsr_dim=371,
                                        #obsr_dim=3630,
                                        #obsr_dim = env.observation_space.shape[0],
                                        device = device
                                       ).to(device)

    # advantage_generator_net= net.Advantage_generator_net(
    #     )

    ######
    # algorithm setup
    ######

    # init replay buffer
    replay_buffer = Buffer(max_size = args.replay_size)
    
    if str.lower(args.alg_name) == 'mql':

        # tdm3 uses specific runner
        from misc.runner_multi_snapshot import Runner
        from algs.MQL.multi_tasks_snapshot import MultiTasksSnapshot
        import algs.MQL.mql as alg

        alg = alg.MQL(actor = actor_net,
                        actor_target = actor_target_net,
                        critic = critic_net,
                        critic_target = critic_target_net,
                        lr = args.lr,
                        gamma=args.gamma,
                        ptau = args.ptau,
                        policy_noise = args.policy_noise,
                        noise_clip = args.noise_clip,
                        policy_freq = args.policy_freq,
                        batch_size = args.batch_size,
                        max_action = max_action,
                        beta_clip = args.beta_clip,
                        prox_coef = args.prox_coef,
                        type_of_training = args.type_of_training,
                        lam_csc = args.lam_csc,
                        use_ess_clipping = args.use_ess_clipping,
                        enable_beta_obs_cxt = args.enable_beta_obs_cxt,
                        use_normalized_beta = args.use_normalized_beta,
                        reset_optims = args.reset_optims,
                        device = device,
                        # advantage_generator_net = advantage_generator_net,
                      )
        ##### rollout/batch generator
        tasks_buffer = MultiTasksSnapshot(max_size = args.snapshot_size)
        rollouts = Runner(env = env,
                          model = alg,
                          replay_buffer = replay_buffer,
                          tasks_buffer = tasks_buffer,
                          burn_in = args.burn_in,
                          expl_noise = args.expl_noise,
                          total_timesteps = args.total_timesteps,
                          max_path_length = args.max_path_length,
                          history_length = args.history_length,
                          device = device)

    else:
        raise ValueError("%s alg is not supported" % args.alg_name)


    ##### rollout/batch generator
    train_tasks, eval_tasks = sample_env_tasks(env, args)

    tasks_buffer.init(train_tasks)
    alg.set_tasks_list(train_tasks)

    print('-----------------------------')
    print("Name of env:", args.env_name)
    print("Tasks:", args.n_tasks )
    print("Train tasks:", args.n_train_tasks  )
    print("Eval tasks:", args.n_eval_tasks)
    print("######### Using Hist len %d #########" % (args.history_length))

    if args.enable_promp_envs == True:
        print("********* Using ProMp Envs *********")
    else:
        print("@@@@@@@@@ Using PEARL Envs @@@@@@@@@")
    print('----------------------------')

    ##############################
    # Train and eval
    #############################
    # define some req vars
    timesteps_since_eval = 0
    episode_num = 0
    update_iter = 5
    sampling_loop = 0

    # episode_stats for raw rewards
    epinfobuf = deque(maxlen=args.n_train_tasks)
    epinfobuf_v2 = deque(maxlen=args.n_train_tasks)

    #todo  just to keep params
    take_snapshot(args, ck_fname_part, alg, 0)
    # alg.save('/home/gzz/Test_ASLAM_LAF_MQL /T_data_model_2' + '/' + str(save_time))
    # save_time += 1
    obs = env.reset()
    rs, ls, ans = evaluate_policy(env, alg, episode_num, update_iter, eparams=args)
    # plt.plot(s, ls, label='linear_speed', color='red', linewidth=1)
    # plt.plot(s, ans, label='angular_speed', color='blue', linewidth=1)
    # plt.legend()  # 显示标签
    # plt.title('a1')  # 设置图片标题
    # plt.xlabel("step")  # 横轴名字
    # plt.ylabel("speed")  # 纵轴名字
    # plt.show()
    # for i in range(2):
    #     episode_num = i
    #     print('-------------episode_num:', episode_num )
    #     step, linear_speed, angular_speed = evaluate_policy(env, alg, episode_num, update_iter, eparams=args)
    #     # loggerr = SummaryWriter(log_dir="data/log4_test")
    #     # loggerr.add_scalar('linear_speed', linear_speed, global_step=step)
    #     # loggerr.add_scalar('angular_speed', angular_speed, global_step=step)
    #     # loggerr.add_scalar('avg_reward', avg_reward, global_step=episode_num)
    #     # loggerr.add_scalar('avg_linear_speed', avg_linear_speed, global_step=episode_num)
    #     # loggerr.add_scalar('avg_angular_speed', avg_angular_speed, global_step=episode_num)
    #     # loggerr.add_scalar('step', step, global_step=episode_num)
    #     i += 1





"""
     # Evaluate untrained policy
    eval_results = [evaluate_policy(env, alg, episode_num, update_iter, etasks=eval_tasks, eparams=args)] 
    if args.enable_train_eval:
        train_subset = np.random.choice(train_tasks, len(eval_tasks))
        train_subset_tasks_eval = evaluate_policy(env, alg, episode_num, update_iter,
                                                  etasks=train_subset,
                                                  eparams=args,
                                                  msg ='Train-Eval')
    else:
        train_subset_tasks_eval = 0

    ## keep track of adapt stats
    if args.enable_adaptation == True:
        args.adapt_csv_hearder =  dict.fromkeys(['eps_num', 'iter','critic_loss', 'actor_loss',
                                                 'csc_samples_neg','csc_samples_pos','train_acc',
                                                 'snap_iter','beta_score','main_critic_loss',
                                                 'main_actor_loss', 'main_beta_score', 'main_prox_critic',
                                                 'main_prox_actor','main_avg_prox_coef',
                                                 'tidx', 'avg_rewards', 'one_raw_reward'])
        adapt_csv_stats = CSVWriter(fname_adapt, args.adapt_csv_hearder)


    # Start total timer
    tstart = time.time()
    ####
    # First fill up the replay buffer with all tasks
    ####
    # max_cold_start = np.maximum(args.num_initial_steps * args.n_train_tasks, args.burn_in)
    # print('Start burnining for at least %d' % max_cold_start)
    keep_sampling = True
    avg_length = 0
    while (keep_sampling == True):

        for idx in range(args.n_train_tasks):
            tidx = train_tasks[idx]
            if args.enable_promp_envs == True:
                env.set_task(tidx) # tidx for promp is task value

            else:
                # for pearl env, tidx == idx
                env.reset_task(tidx) # tidx here is an id

            data = rollouts.run(update_iter, keep_burning = True, task_id=tidx,
                                early_leave = args.max_path_length/4) # data collection is way important now
            timesteps_since_eval += data['episode_timesteps']
            update_iter += data['episode_timesteps']
            epinfobuf.extend(data['epinfos'])
            epinfobuf_v2.extend(data['epinfos'])
            episode_num += 1
            avg_length += data['episode_timesteps']
            if update_iter >= max_cold_start:
                keep_sampling = False
                break
    #TODO bug
    print('There are %d samples in buffer now' % replay_buffer.size_rb())
    print('Average length %.2f for %d episode_nums for %d max_cold_start steps' % (avg_length/episode_num, episode_num, max_cold_start))
    print('Episode_nums/tasks %.2f and avg_len/tasks %.2f ' % (episode_num/args.n_train_tasks, avg_length/args.n_train_tasks))
    avg_epi_length = int(avg_length/episode_num)
    # already seen all tasks once
    sampling_loop = 1

    ####
    # Train and eval main loop
    ####
    train_iter = 0 
    lr_updated = False
    while update_iter < args.total_timesteps:
        if args.enable_promp_envs:
            train_tasks = env.sample_tasks(args.n_train_tasks)
            train_indices = train_tasks.copy()

        else:
            #shuffle the ind
            train_indices = np.random.choice(train_tasks, len(train_tasks))

        for tidx in train_indices:
            ######
            # update learning rate
            ######
            if args.lr_milestone > -1 and lr_updated == False and update_iter > args.lr_milestone:
                update_lr(args, update_iter, alg)
                lr_updated = True

            #######
            # run training to calculate loss, run backward, and update params
            #######
            stats_csv = None

            #adjust training steps
            adjusted_no_steps = adjust_number_train_iters(buffer_size = replay_buffer.size_rb(),
                                     num_train_steps = args.num_train_steps,
                                     bsize = args.batch_size,
                                     min_buffer_size = args.min_buffer_size,
                                     episode_timesteps = avg_epi_length,
                                     use_epi_len_steps = args.use_epi_len_steps)

            alg_stats, stats_csv = alg.train(replay_buffer = replay_buffer,
                                      iterations = adjusted_no_steps,
                                      tasks_buffer = tasks_buffer,
                                      train_iter = train_iter,
                                      task_id = tidx
                                      )
            train_iter += 1
            #######
            # logging
            #######
            nseconds = time.time() - tstart
            # Calculate the fps (frame per second)
            fps = int(( update_iter) / nseconds)

            if ((episode_num % args.log_interval == 0 or episode_num % len(train_tasks)/2 == 0) or episode_num == 1 ):
                loggerr.add_scalar("episode_reward", float(data['episode_reward']), global_step=episode_num)
                loggerr.add_scalar("critic_loss",float(alg_stats['critic_loss']),global_step=episode_num)
                loggerr.add_scalar("actor_loss", float(alg_stats['actor_loss']), global_step=episode_num)
                loggerr.add_scalar('nupdates', update_iter, global_step=episode_num)
                loggerr.add_scalar('fps', fps, global_step=episode_num)
                loggerr.add_scalar('total_timesteps', update_iter, global_step=episode_num)
                loggerr.add_scalar('eprewmean', float(safemean([epinfo['r'] for epinfo in epinfobuf])),
                                   global_step=episode_num)
                loggerr.add_scalar('eplenmean', float(safemean([epinfo['l'] for epinfo in epinfobuf])),
                                   global_step=episode_num)
                loggerr.add_scalar('linear_speed', float(safemean([epinfo['x'] for epinfo in epinfobuf])),
                                   global_step=episode_num)
                loggerr.add_scalar('angular_speed', float(safemean([epinfo['z'] for epinfo in epinfobuf])),
                                   global_step=episode_num)
                loggerr.add_scalar('sampling_loop', sampling_loop, global_step=episode_num)
                loggerr.add_scalar('buffer_size', replay_buffer.size_rb(), global_step=episode_num)
                loggerr.add_scalar('adjusted_no_steps', adjusted_no_steps, global_step=episode_num)
                #loggerr.add_scalar('avg_reward', float(data['avg_reward'], global_step=episode_num)

            #     logger.record_tabular("nupdates", update_iter)
            #     logger.record_tabular("fps", fps)
            #     logger.record_tabular("total_timesteps", update_iter)
            #     logger.record_tabular("critic_loss", float(alg_stats['critic_loss']))
            #     logger.record_tabular("actor_loss", float(alg_stats['actor_loss']))
            #     logger.record_tabular("episode_reward", float(data['episode_reward']))
            #     logger.record_tabular('eprewmean', float(safemean([epinfo['r'] for epinfo in epinfobuf])))
            #     logger.record_tabular('eplenmean', float(safemean([epinfo['l'] for epinfo in epinfobuf])))
            #     logger.record_tabular("episode_num", episode_num)
            #     logger.record_tabular("sampling_loop", sampling_loop)
            #     logger.record_tabular("buffer_size", replay_buffer.size_rb())
            #     logger.record_tabular("adjusted_no_steps", adjusted_no_steps)
            # #
            #     if 'actor_mmd_loss' in alg_stats:
            #         logger.record_tabular("critic_mmd_loss", float(alg_stats['critic_mmd_loss']))
            #         logger.record_tabular("actor_mmd_loss", float(alg_stats['actor_mmd_loss']))
            #
            #     if 'beta_score' in alg_stats:
            #          logger.record_tabular("beta_score", float(alg_stats['beta_score']))
            #
            #     logger.dump_tabular()
            #     print(("Total T: %d Episode Num: %d Episode Len: %d Reward: %f") %
            #           (update_iter, episode_num, data['episode_timesteps'], data['episode_reward']))
            #
            #     #print out some info about CSC
            # if stats_csv:
            #         print(("CSC info:  critic_loss: %.4f actor_loss: %.4f No beta_score: %.4f ") %
            #               (stats_csv['critic_loss'], stats_csv['actor_loss'], stats_csv['beta_score']))
            #         if 'csc_info' in stats_csv:
            #             print(("Number of examples used for CSC, prediction accuracy on train, and snap Iter: single: %d multiple tasks: %d  acc: %.4f snap_iter: %d ") %
            #                 (stats_csv['csc_info'][0], stats_csv['csc_info'][1], stats_csv['csc_info'][2], stats_csv['snap_iter']))
            #             print(("Prox info: prox_critic %.4f prox_actor: %.4f")%(alg_stats['prox_critic'], alg_stats['prox_actor']))
            #
            #         if 'avg_prox_coef' in alg_stats and 'csc_info' in stats_csv:
            #             print(("\ravg_prox_coef: %.4f" %(alg_stats['avg_prox_coef'])))

            #######
            # run eval
            #######
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq

                if args.enable_adaptation == True:
                    eval_temp = evaluate_policy(env, alg, episode_num, update_iter,
                                                etasks=eval_tasks, eparams=args,
                                                meta_learner = alg,
                                                train_tasks_buffer = tasks_buffer,
                                                train_replay_buffer = replay_buffer)

                else:
                    eval_temp = evaluate_policy(env, alg, episode_num, update_iter, etasks=eval_tasks, eparams=args)

                eval_results.append(eval_temp)

                # Eval subset of train tasks
                if args.enable_train_eval:

                    if args.enable_promp_envs == False:
                        train_subset = np.random.choice(train_tasks, len(eval_tasks))

                    else:
                        train_subset = None

                    train_subset_tasks_eval = evaluate_policy(env, alg, episode_num, update_iter,
                                                              etasks=train_subset,
                                                              eparams=args,
                                                              msg ='Train-Eval')
                else:
                    train_subset_tasks_eval = 0

                # dump results
                # wrt_csv_eval.write({'nupdates':update_iter,
                #                    'total_timesteps':update_iter,
                #                    'eval_eprewmean':eval_temp,
                #                    'train_eprewmean':train_subset_tasks_eval,
                #                    'episode_num':episode_num,
                #                    'sampling_loop':sampling_loop})

            #######
            # todo save for every interval-th episode or for the last epoch
            #######
            if (episode_num % args.save_freq == 0 or episode_num == args.total_timesteps - 1):
                    take_snapshot(args, ck_fname_part, alg, update_iter)
                    alg.save('/home/gzz/Test_ASLAM_LAF_MQL /T_data_model' + '/' +str(save_time))
                    save_time += 1

            #######
            # Interact and collect data until reset
            #######
            # should reset the queue, as new trail starts
            epinfobuf = deque(maxlen=args.n_train_tasks)
            avg_epi_length = 0

            for sl in range(args.num_tasks_sample):

                if sl > 0:
                    idx = np.random.randint(len(train_tasks))
                    tidx = train_tasks[idx]

                if args.enable_promp_envs == True:
                    env.set_task(tidx) # tidx for promp is task value

                else:
                    env.reset_task(tidx) # tidx here is an id

                data = rollouts.run(update_iter, task_id = tidx)
                timesteps_since_eval += data['episode_timesteps']
                update_iter += data['episode_timesteps']
                epinfobuf.extend(data['epinfos'])
                epinfobuf_v2.extend(data['epinfos'])
                episode_num += 1
                avg_epi_length += data['episode_timesteps']

            avg_epi_length = int(avg_epi_length/args.num_tasks_sample)

        # just to keep track of how many times all training tasks have been seen
        sampling_loop += 1

    ###############
    # Eval for the final time
    ###############
    eval_temp = evaluate_policy(env, alg, episode_num, update_iter, etasks=eval_tasks, eparams=args)
    # Eval subset of train tasks
    if args.enable_promp_envs == False:
        train_subset = np.random.choice(train_tasks, len(eval_tasks))

    else:
        train_subset = None

    train_subset_tasks_eval = evaluate_policy(env, alg, episode_num, update_iter,
                                              etasks=train_subset,
                                              eparams=args,
                                              msg ='Train-Eval')

    eval_results.append(eval_temp)
    # wrt_csv_eval.write({'nupdates':update_iter,
    #                    'total_timesteps':update_iter,
    #                    'eval_eprewmean':eval_temp,
    #                    'train_eprewmean':train_subset_tasks_eval,
    #                    'episode_num':episode_num,
    #                    'sampling_loop':sampling_loop})
    # wrt_csv_eval.close()
    print('Done')"""