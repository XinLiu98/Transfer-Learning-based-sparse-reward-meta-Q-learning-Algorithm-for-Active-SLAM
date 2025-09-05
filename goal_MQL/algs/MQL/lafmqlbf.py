from __future__ import  print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from copy import deepcopy
from sklearn.linear_model import LogisticRegression as logistic
from models.networks import FlattenMlp
import copy
from torch.optim import Adam
from rlkit.discor.utils import disable_gradients, soft_update, update_params

class MQL:

    def __init__(self, 
                actor,
                actor_target,
                critic,
                advantage_generator_net,
                critic_target,
                lr=None,
                #state_dim=3634,
                #TODO 363
                state_dim=363,
                action_dim=2,
                gamma=0.99,
                ptau = 0.005,
                policy_noise = 0.2,
                noise_clip = 0.5,
                policy_freq = 2,
                batch_size = 100,
                optim_method = '',
                max_action = None,
                max_iter_logistic = 2000,
                beta_clip = 1,
                enable_beta_obs_cxt = False,
                prox_coef = 1,
                device = 'cpu',
                lam_csc = 1.0,
                type_of_training = 'csc',
                use_ess_clipping = False,
                use_normalized_beta = True,
                reset_optims = False,
                learn_advantage_function_inner = True,
                td3 = True,
                #advantage_generator = True,
                adv_weights = None,
                alpha=2.5,
                lfiw =True,
                discor= True,
                tau_init=10.0,
                tau_scale=1,
                hard_tper_weight=0.4,
                tper=False,
                eval_tper=True,
                use_backward_timestep=False,
                reweigh_type="hard",
                reweigh_hyper=None,
                prob_hidden_units=[50, 50],
                q_lr=0.0003,
                prob_temperature=7.5,
                 ):
    # construct_network,
        '''
            actor:  actor network 
            critic: critic network 
            lr:   learning rate for RMSProp
            gamma: reward discounting parameter
            ptau:  Interpolation factor in polyak averaging  
            policy_noise: add noise to policy 
            noise_clip: clipped noise 
            policy_freq: delayed policy updates
            enable_beta_obs_cxt:  decide whether to concat obs and ctx for logistic regresstion
            lam_csc: logisitc regression reg, samller means stronger reg
        '''
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.advantage_generator_net = advantage_generator_net
        self.gamma = gamma
        self.ptau = ptau
        self.policy_noise = policy_noise
        self.policy_freq  = policy_freq
        self.noise_clip = noise_clip
        self.max_action = max_action
        self.batch_size = batch_size
        self.max_iter_logistic = max_iter_logistic
        self.beta_clip = beta_clip
        self.enable_beta_obs_cxt = enable_beta_obs_cxt
        self.prox_coef = prox_coef
        self.prox_coef_init = prox_coef
        self.device = device
        self.lam_csc = lam_csc
        self.type_of_training = type_of_training
        self.use_ess_clipping = use_ess_clipping
        self.r_eps = np.float32(1e-7)  # this is used to avoid inf or nan in calculations
        self.use_normalized_beta = use_normalized_beta
        self.set_training_style()
        self.lr = lr
        self.reset_optims = reset_optims
        """.xin"""
        self.learn_advantage_function_inner = learn_advantage_function_inner
        self.td3 = td3
        self.adv_weights = adv_weights

        self.alpha = alpha
        self.lfiw = lfiw
        self.discor = discor
        self.tper = tper
        print("----------------lfiw------------------",self.lfiw)

        # TODO torch.load qudiao 3.14

        # state_dict = torch.load('/home/gzz/data/ck/gazebo-dir_mql_dummy.pt')
        #


        state_dict = torch.load('/home/gzz/goal_MQL/ck/ck_07_00/gazebo-dir_mql_dummy.pt')

        # state_dict = torch.load('/home/gzz/goal_MQL/ck/discor/lfiw/gazebo-dir_mql_dummy.pt')
        self.actor.load_state_dict(state_dict['model_states_actor'])
        self.critic.load_state_dict(state_dict['model_states_critic'])
        #
        # for key in state_dict["model_states_actor"].items():
        #     print(key, sep="   ")
        # for key in state_dict["model_states_critic"].items():
        #     print(key, sep="   ")

    # load tragtes models.
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # keep a copy of model params which will be used for proximal point
        self.copy_model_params()

        if lr:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr)

        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters())
            self.critic_optimizer = optim.Adam(self.critic.parameters())
        """xin"""


        print('-----------------------------')
        print('Optim Params')
        print("Actor:\n ",  self.actor_optimizer)
        print("Critic:\n ", self.critic_optimizer )
        print('********')
        print("reset_optims: ", reset_optims)
        print("use_ess_clipping: ", use_ess_clipping)
        print("use_normalized_beta: ", use_normalized_beta)
        print("enable_beta_obs_cxt: ", enable_beta_obs_cxt)
        print('********')
        print('-----------------------------')

        if self.discor:
            self._tau1 = torch.tensor(
                tau_init, device=self.device, requires_grad=False)
            self._tau2 = torch.tensor(
                tau_init, device=self.device, requires_grad=False)

            if tau_init < 1e-6:
                self.no_tau = True
                print("===========No tau!==========")
            else:
                self.no_tau = False
            self.tau_scale = tau_scale

        if self.lfiw:
            self._prob_classifier = FlattenMlp(
                input_size=373,#374 , #state_dim + action_dim,
                output_size=1,
                hidden_sizes=prob_hidden_units,
            ).to(device=self.device)
            self._prob_optim = Adam(
                self._prob_classifier.parameters(), lr=q_lr)
            self.prob_temperature = prob_temperature

        if self.tper:
            self.hard_tper_weight = hard_tper_weight
            self.use_backward_timestep = use_backward_timestep
            self.reweigh_type = reweigh_type
            self.reweigh_hyper = reweigh_hyper
            self.l, self.h, self.k, self.b = \
                [torch.tensor(i).to(device=self.device) for i in self.reweigh_hyper["linear"]]
            if self.reweigh_type in ["adaptive_linear", "done_cnt_linear"]:
                self.low_l, self.low_h, self.high_l, self.high_h, self.t_s, self.t_e = \
                    [torch.tensor(i).to(device=self.device) for i in self.reweigh_hyper["adaptive_linear"]]
            if "exp" in self.reweigh_type:
                self.exp_k, self.exp_gamma = self.reweigh_hyper["exp"]

    def _configure(self):
        """Initializes variables.xin"""
        self.network_generator = self.config.network_generator
        self.value_network_generator = self.config.value_network_generator
        self.task_generator = self.config.task_generator
        self.learn_advantage_function_inner = self.config.learn_advantage_function_inner
        if self.config.advantage_generator:
            self.advantage_generator = self.config.advantage_generator
        else:
            self.advantage_generator = None
        if self.config.td3:
            self.td3 = self.config.td3
        else:
            self.td3 = False


    def copy_model_params(self):
        '''
            Keep a copy of actor and critic for proximal update
        '''
        self.ckpt = {
                        'actor': deepcopy(self.actor),
                        'critic': deepcopy(self.critic)
                    }

    def set_tasks_list(self, tasks_idx):
        '''
            Keep copy of task lists
        '''
        self.train_tasks_list = set(tasks_idx.copy())


   # def select_action(self, obs, previous_action, previous_reward, previous_obs, state, images, transformation, num_pre):
    def select_action(self, obs, previous_action, previous_reward, previous_obs):
        '''
            return action
        '''

        print('mql_161',obs.shape)
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        previous_action = torch.FloatTensor(previous_action.reshape(1, -1)).to(self.device)
        previous_reward = torch.FloatTensor(previous_reward.reshape(1, -1)).to(self.device)
        if previous_obs.shape==[1, 3]:
            previous_obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        else:
            previous_obs = torch.FloatTensor(previous_obs.reshape(1, -1)).to(self.device)
            #print('mql_169', previous_obs.shape)

        # combine all other data here before send them to actor
        # torch.cat([previous_action, previous_reward], dim = -1)
        pre_act_rew = [previous_action, previous_reward, previous_obs]
        #print('mql178', previous_action.shape, previous_obs.shape, previous_reward.shape)
        #print('mql_173')


        return self.actor(obs, pre_act_rew).cpu().data.numpy().flatten()

    def get_prox_penalty(self, model_t, model_target):
        '''
            This function calculates ||theta - theta_t||
        '''
        param_prox = []
        for p, q in zip(model_t.parameters(), model_target.parameters()):
            # q should ne detached
            param_prox.append((p - q.detach()).norm()**2)

        result = sum(param_prox)

        return result

    def train_cs(self, task_id = None, snap_buffer = None, train_tasks_buffer = None, adaptation_step = False):
        '''
            This function trains covariate shift correction model
        '''

        ######
        # fetch all_data
        ######
        if adaptation_step == True:
            # step 1: calculate how many samples per classes we need
            # in adaption step, all train task can be used
            task_bsize = int(snap_buffer.size_rb(task_id) / (len(self.train_tasks_list))) + 2
            neg_tasks_ids = self.train_tasks_list

        else:
            # step 1: calculate how many samples per classes we need
            task_bsize = int(snap_buffer.size_rb(task_id) / (len(self.train_tasks_list) - 1)) + 2
            neg_tasks_ids = list(self.train_tasks_list.difference(set([task_id])))
        #todo x,y

        # collect examples from other tasks and consider them as one class
        # view --> len(neg_tasks_ids),task_bsize, D ==> len(neg_tasks_ids) * task_bsize, D
        pu, pr, px, xx = train_tasks_buffer.sample(task_ids = neg_tasks_ids, batch_size = task_bsize)
        neg_actions = torch.FloatTensor(pu).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_rewards = torch.FloatTensor(pr).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_obs = torch.FloatTensor(px).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_xx = torch.FloatTensor(xx).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)

        # sample cuurent task and consider it as another class
        # returns size: (task_bsize, D)
        ppu, ppr, ppx, pxx = snap_buffer.sample(task_ids = [task_id], batch_size = snap_buffer.size_rb(task_id))
        pos_actions = torch.FloatTensor(ppu).to(self.device)
        pos_rewards = torch.FloatTensor(ppr).to(self.device)
        pos_obs = torch.FloatTensor(ppx).to(self.device)
        pos_pxx = torch.FloatTensor(pxx).to(self.device)

        # combine reward and action and previous states for context network.
        pos_act_rew_obs  = [pos_actions, pos_rewards, pos_obs]
        neg_act_rew_obs  = [neg_actions, neg_rewards, neg_obs]

        ######
        # extract features: context features 
        ######
        with torch.no_grad():   

            # batch_size X context_hidden 
            # self.actor.get_conext_feats outputs, [batch_size , context_size]
            # torch.cat ([batch_size , obs_dim], [batch_size , context_size]) ==> [batch_size, obs_dim + context_size ]
            if self.enable_beta_obs_cxt == True:
                snap_ctxt = torch.cat([pos_pxx, self.actor.get_conext_feats(pos_act_rew_obs)], dim = -1).cpu().data.numpy()
                neg_ctxt = torch.cat([neg_xx, self.actor.get_conext_feats(neg_act_rew_obs)], dim = -1).cpu().data.numpy()

            else:
                snap_ctxt = self.actor.get_conext_feats(pos_act_rew_obs).cpu().data.numpy()
                neg_ctxt = self.actor.get_conext_feats(neg_act_rew_obs).cpu().data.numpy()


        ######
        # Train logistic classifiers 
        ######
        x = np.concatenate((snap_ctxt, neg_ctxt)) # [b1 + b2] X D
        y = np.concatenate((-np.ones(snap_ctxt.shape[0]), np.ones(neg_ctxt.shape[0])))

        # model params : [1 , D] wehere D is context_hidden
        model = logistic(solver='lbfgs', max_iter = self.max_iter_logistic, C = self.lam_csc).fit(x,y)
        # keep track of how good is the classifier
        predcition_score = model.score(x, y)

        info = (snap_ctxt.shape[0], neg_ctxt.shape[0],  model.score(x, y))
        #print(info)
        return model, info

    def update_prox_w_ess_factor(self, cs_model, x, beta=None):
        '''
            This function calculates effective sample size (ESS):
            ESS = ||w||^2_1 / ||w||^2_2  , w = pi / beta
            ESS = ESS / n where n is number of samples to normalize
            x: is (n, D)
        '''
        n = x.shape[0]
        if beta is not None:
            # beta results should be same as using cs_model.predict_proba(x)[:,0] if no clipping
            w = ((torch.sum(beta)**2) /(torch.sum(beta**2) + self.r_eps) )/n
            ess_factor = np.float32(w.numpy())

        else:
            # step 1: get prob class 1
            p0 = cs_model.predict_proba(x)[:,0]
            w =  p0 / ( 1 - p0 + self.r_eps)
            w = (np.sum(w)**2) / (np.sum(w**2) + self.r_eps)
            ess_factor = np.float32(w) / n

        # since we assume task_i is class -1, and replay buffer is 1, then
        ess_prox_factor = 1.0 - ess_factor

        if np.isnan(ess_prox_factor) or np.isinf(ess_prox_factor) or ess_prox_factor <= self.r_eps: # make sure that it is valid
            self.prox_coef = self.prox_coef_init

        else:
            self.prox_coef = ess_prox_factor

    def get_propensity(self, cs_model, curr_pre_act_rew, curr_obs):
        '''
            This function returns propensity for current sample of data 
            simply: exp(f(x))
        '''

        ######
        # extract features: context features 
        ######
        with torch.no_grad():

            # batch_size X context_hidden 
            if self.enable_beta_obs_cxt == True:
                ctxt = torch.cat([curr_obs, self.actor.get_conext_feats(curr_pre_act_rew)], dim = -1).cpu().data.numpy()

            else:
                ctxt = self.actor.get_conext_feats(curr_pre_act_rew).cpu().data.numpy()

        # step 0: get f(x)
        f_prop = np.dot(ctxt, cs_model.coef_.T) + cs_model.intercept_

        # step 1: convert to torch
        f_prop = torch.from_numpy(f_prop).float()

        # To make it more stable, clip it
        f_prop = f_prop.clamp(min=-self.beta_clip)

        # step 2: exp(-f(X)), f_score: N * 1
        f_score = torch.exp(-f_prop)
        f_score[f_score < 0.1]  = 0 # for numerical stability

        if self.use_normalized_beta == True:

            #get logistic regression prediction of class [-1] for current task
            lr_prob = cs_model.predict_proba(ctxt)[:,0]
            # normalize using logistic_probs
            d_pmax_pmin = np.float32(np.max(lr_prob) - np.min(lr_prob))
            f_score = ( d_pmax_pmin * (f_score - torch.min(f_score)) )/( torch.max(f_score) - torch.min(f_score) + self.r_eps ) + np.float32(np.min(lr_prob))

        # update prox coeff with ess.
        if self.use_ess_clipping == True:
            self.update_prox_w_ess_factor(cs_model, ctxt, beta=f_score)


        return f_score, None

    def calc_update_d_pi_iw(self, slow_obs, slow_act, fast_obs, fast_act, target_obs=None, target_act=None):
        print('slow_obs', slow_obs.shape, slow_obs.size())
        print('slow_act', slow_act.shape, slow_act.size())
        print('fast_obs', fast_obs.shape, fast_obs.size())
        print('fast_act', fast_act.shape, fast_act.size())
        # f1 = torch.zeros(400, 7087)

        slow_obss = slow_obs[:,0:371]
        slow_actt = slow_act[:,0:2]
        print('slow_obss', slow_obss.shape, slow_obss.size())
        print('slow_actt', slow_actt.shape, slow_actt.size())
        slow_samples = torch.cat((slow_obss, slow_actt), dim=1)
        fast_samples = torch.cat((fast_obs, fast_act), dim=1)


        zeros = torch.zeros(slow_samples.size(0)).to(device=self.device)
        ones = torch.ones(fast_samples.size(0)).to(device=self.device)

        slow_preds = self._prob_classifier(slow_samples)
        fast_preds = self._prob_classifier(fast_samples)

        slow_preds = slow_preds.squeeze(-1)
        fast_preds = fast_preds.squeeze(-1)

        loss = F.binary_cross_entropy(F.sigmoid(slow_preds), zeros) + \
                F.binary_cross_entropy(F.sigmoid(fast_preds), ones)

        update_params(self._prob_optim, loss)

        # In case we want to compute ratio on data different from what we train the network
        if target_obs is None:
            target_obs = slow_obss
        if target_act is None:
            target_act = slow_actt
        # target_act = torch.tensor(target_act)
        # target_obs = target_obs.squeeze(1)
        # print('target_obs',target_obs.size())
        # print('target_act',target_act.size())
        target_obss = target_obs[:, 0:371]
        target_actt = target_act[:, 0:2]
        target_samples = torch.cat((target_obss, target_actt), dim=1)
        slow_preds = self._prob_classifier(target_samples)

        importance_weights = F.sigmoid(slow_preds/self.prob_temperature).detach()
        importance_weights = importance_weights / torch.sum(importance_weights)

        return importance_weights, loss

    def do_training(self,
                    replay_buffer = None,
                    iterations = None,
                    csc_model = None,
                    apply_prox = False,
                    current_batch_size = None,
                    src_task_ids = []):

        '''
            inputs:
                replay_buffer
                iterations episode_timesteps                 
        '''
        print('------do_training------')
        actor_loss_out = 0.0
        critic_loss_out = 0.0
        critic_prox_out = 0.0
        actor_prox_out = 0.0
        list_prox_coefs = [self.prox_coef]

        for it in range(iterations):

            ########
            # Sample replay buffer 
            ########
            if len(src_task_ids) > 0:
                x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample_tasks(task_ids = src_task_ids, batch_size = current_batch_size)

            else:
                x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample(current_batch_size)

            obs = torch.FloatTensor(x).to(self.device)
            next_obs = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            doness = torch.FloatTensor(d).to(self.device)
            mask = torch.FloatTensor(1 - d).to(self.device)
            previous_action = torch.FloatTensor(pu).to(self.device)
            previous_reward = torch.FloatTensor(pr).to(self.device)
            previous_obs = torch.FloatTensor(px).to(self.device)
            print('obs',obs.size())
            print('next_obs', next_obs.size())
            print('previous_obs', previous_obs.size())

            # list of hist_actions and hist_rewards which are one time ahead of previous_ones
            # example:
            # previous_action = [t-3, t-2, t-1]
            # hist_actions    = [t-2, t-1, t]
            hist_actions = torch.FloatTensor(nu).to(self.device)
            hist_rewards = torch.FloatTensor(nr).to(self.device)
            hist_obs     = torch.FloatTensor(nx).to(self.device)
            print('hist_obs',hist_obs.size())

            # combine reward and action
            act_rew = [hist_actions, hist_rewards, hist_obs] # torch.cat([action, reward], dim = -1)
            act_rew_tensor = torch.cat([hist_actions, hist_rewards, hist_obs], dim=-1)
            pre_act_rew = [previous_action, previous_reward, previous_obs] #torch.cat([previous_action, previous_reward], dim = -1)
            pre_act_rew_tensor = torch.cat([previous_action, previous_reward, previous_obs], dim=-1)

            # batch = {
            #     'states': obs,
            #     'actions': action,
            #     'rewards': act_rew,
            #     'dones': mask,
            #     'next_states': next_obs
            # }


            if csc_model is None:
                # propensity_scores dim is batch_size
                # no csc_model, so just do business as usual
                beta_score = torch.ones((current_batch_size, 1)).to(self.device)

            else:
                # propensity_scores dim is batch_size
                beta_score, clipping_factor = self.get_propensity(csc_model, pre_act_rew, obs)
                beta_score = beta_score.to(self.device)
                list_prox_coefs.append(self.prox_coef)

            ########
            # Select action according to policy and add clipped noise 
            # mu'(s_t) = mu(s_t | \theta_t) + N (Eq.7 in https://arxiv.org/abs/1509.02971) 
            # OR
            # Eq. 15 in TD3 paper:
            # e ~ clip(N(0, \sigma), -c, c)
            ########
            noise = (torch.randn_like(action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs, act_rew) + noise).clamp(-self.max_action, self.max_action)
            print('###mql421next_action')
            next_action_tensor = torch.Tensor(next_action)

            if self.lfiw:
                # fast_batch = batch['fast']
                # next_obss = torch.squeeze(next_obs, 1)
                obss = torch.squeeze(obs, 1)
                fast_states, fast_actions = obss, next_action_tensor  # act_rew_tensor #fast_batch['states'], fast_batch['actions']
            else:
                fast_batch = None
                # uniform_batch = batch["uniform"]
            slow_states, slow_actions = previous_obs, previous_action#hist_obs, hist_actions #pre_act_rew_tensor  # uniform_batch["states"], uniform_batch["actions"]
            # train_batch = batch["uniform"]
            states, actions, next_states, dones = previous_obs, previous_action, next_obs, doness
            # train_batch["states"], train_batch["actions"], train_batch["next_states"], train_batch["dones"]
            # s,a to update the weight of lfiw network
            batch_size = states.shape[0]
            weights1 = torch.ones((batch_size, 1)).to(device=self.device)
            weights2 = torch.ones((batch_size, 1)).to(device=self.device)
            if self.discor:
                # discor_weights = self.calc_importance_weights(next_states, dones)
                with torch.no_grad():
                    next_actions = (self.actor_target(next_obs, act_rew) + noise).clamp(-self.max_action,
                                                                                        self.max_action)
                    next_errs1, next_errs2 = self.critic_target(next_obs, next_action, act_rew)

                # Terms inside the exponent of importance weights.
                if self.no_tau:
                    x1 = -(1.0 - dones) * 0.99 * next_errs1
                    x2 = -(1.0 - dones) * 0.99 * next_errs2
                else:
                    x1 = -(1.0 - dones) * 0.99 * next_errs1 / (self._tau1 * self.tau_scale)
                    x2 = -(1.0 - dones) * 0.99 * next_errs2 / (self._tau2 * self.tau_scale)

                # Calculate self-normalized importance weights.
                imp_ws1 = F.softmax(x1, dim=0)
                imp_ws2 = F.softmax(x2, dim=0)
                # print(weights[0].shape, discor_weights[0].shape)
                weights1 *= imp_ws1 #discor_weights[0]
                weights2 *= imp_ws2 #discor_weights[1]

            if self.lfiw:
                lfiw_weights, prob_loss = self.calc_update_d_pi_iw(slow_states, slow_actions, fast_states, fast_actions,
                                                                   states, actions)
                weights1 *= lfiw_weights
                weights2 *= lfiw_weights

            # if self.tper:
            #     steps = train_batch["steps"]
            #     done_cnts = done#train_batch["done_cnts"]
            #     tper_weights = self.calc_tper_weights(steps, done_cnts)
            #     weights1 *= tper_weights
            #     weights2 *= tper_weights
            ########
            #  Update critics
            #  1. Compute the target Q value 
            #  2. Get current Q estimates
            #  3. Compute critic loss
            #  4. Optimize the criticcritic
            ########

            # 1. y = r + \gamma * min{Q1, Q2} (s_next, next_action)
            # if done , then only use reward otherwise reward + (self.gamma * target_Q)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, act_rew)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (mask * self.gamma * target_Q).detach()

            # 2.  Get current Q estimates
            current_Q1, current_Q2 = self.critic(obs, action, pre_act_rew)


            # 3. Compute critic loss
            # even we picked min Q, we still need to backprob to both Qs     critic_loss_temp=TD_error=adv
            critic_loss_temp = F.mse_loss(current_Q1, target_Q, reduction='none') * weights1 + F.mse_loss(current_Q2, target_Q, reduction='none') * weights2
            assert critic_loss_temp.shape == beta_score.shape, ('shape critic_loss_temp and beta_score shoudl be the same',
                                                                critic_loss_temp.shape,
                                                                beta_score.shape)

            #TODO xinzeng 538-542
            adv_input = torch.cat((act_rew_tensor, pre_act_rew_tensor, next_action_tensor),dim=1)
            #train_advantages = self.advantage_generator.construct_network(
            train_advantages = self.advantage_generator_net(
                adv_input
                )
            #train_advantages_new = tfpyth.torch_from_tensorflow( [adv_input, self.adv_weights], train_advantages).apply
            # critic_loss = ( beta_score * train_advantages ).mean()
            # critic_loss = (weights1 * train_advantages).mean()
            critic_loss = (critic_loss_temp * beta_score).mean() #yuan
            critic_loss_out += critic_loss.item()

            if apply_prox:
                # calculate proximal term
                critic_prox = self.get_prox_penalty(self.critic, self.ckpt['critic'])
                critic_loss = critic_loss + self.prox_coef * critic_prox
                critic_prox_out += critic_prox.item()

            # 4. Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ########
            # Delayed policy updates
            ########
            if it % self.policy_freq == 0:

                # Compute actor loss
                # todo td3+bc
                '''
                Q = self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew)
                lmbda = self.alpha / Q.abs().mean().detach()
                pi = self.actor(obs, pre_act_rew)
                #
                actor_loss_temp = -1 * beta_score * lmbda * self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew).mean() + F.mse_loss(pi, action)
                '''

                actor_loss_temp = -1 * beta_score * self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew) #12.6 qu diao
                actor_loss = actor_loss_temp.mean()
                actor_loss_out += actor_loss.item()

                if apply_prox:
                    # calculate proximal term
                    actor_prox = self.get_prox_penalty(self.actor, self.ckpt['actor'])
                    actor_loss = actor_loss + self.prox_coef * actor_prox
                    actor_prox_out += actor_prox.item()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

                # if self.discor:
                    # soft_update(
                        # self._target_error_net, self._online_error_net,
                        # self._target_update_coef)

        out = {}
        if iterations == 0:
            out['critic_loss'] = 0
            out['actor_loss']  = 0
            out['prox_critic'] = 0
            out['prox_actor']  = 0
            out['beta_score']  = 0

        else:
            out['critic_loss'] = critic_loss_out/iterations
            out['actor_loss']  = self.policy_freq * actor_loss_out/iterations
            out['prox_critic'] = critic_prox_out/iterations
            out['prox_actor']  = self.policy_freq * actor_prox_out/iterations
            out['beta_score']  = 1#beta_score.cpu().data.numpy().mean()

        #if csc_model and self.use_ess_clipping == True:
        out['avg_prox_coef'] = np.mean(list_prox_coefs)

        return out

    def calc_tper_weights(self, steps, done_cnts):
        steps = steps.to(dtype=torch.float32)
        rel_step = steps / torch.max(steps)
        if self.use_backward_timestep:
            # convert bk step to forward
            rel_step = 1 - rel_step

        if self.reweigh_type == 'hard':
            assert self.hard_tper_weight <= 0.5
            med = torch.median(steps)
            one = torch.tensor(1-self.hard_tper_weight, device=self.device, requires_grad=False)
            zero = torch.tensor(self.hard_tper_weight, device=self.device, requires_grad=False)
            cond = steps < med if self.use_backward_timestep else steps > med
            weight = torch.where(cond, one, zero)
        elif self.reweigh_type == 'linear':
            weight = self._calc_linear_weight(rel_step, self.l, self.h, self.k, self.b)
        elif self.reweigh_type == 'adaptive_linear':
            cur_low = torch.clamp(
                self.low_l + (self.low_h - self.low_l)/(self.t_e - self.t_s)*(self._learning_steps - self.t_s),
                self.low_l,
                self.low_h
            )
            cur_high = torch.clamp(
                self.high_h + (self.high_l - self.high_h)/(self.t_e - self.t_s)*(self._learning_steps - self.t_s),
                self.high_l,
                self.high_h
            )
            weight = self._calc_linear_weight(rel_step, cur_low, cur_high, self.k, self.b)
        elif self.reweigh_type == 'done_cnt_linear':
            rel_done_cnt = done_cnts.to(dtype=torch.float32) / torch.max(done_cnts)
            # The tajectory is newer with larger done counts, which can be understood as fewer learning steps
            pseudo_step = 1 - rel_done_cnt
            cur_low = torch.clamp(
                self.low_l + (self.low_h - self.low_l) * pseudo_step,
                self.low_l,
                self.low_h
            )
            cur_high = torch.clamp(
                self.high_h + (self.high_l - self.high_h) * pseudo_step,
                self.high_l,
                self.high_h
            )
            weight = self._calc_linear_weight(rel_step, cur_low, cur_high, self.k, self.b)
        elif self.reweigh_type == 'exp':
            # compute exp weight with exp(k*gamma^bk_step)
            if not self.use_backward_timestep:
                # compute proxy backward step
                steps = torch.max(steps) - steps
            weight = torch.exp(self.exp_k * self.exp_gamma ** steps) / (2.71828 - 1)
            weight = weight / torch.sum(weight) * steps.shape[0]
        return weight

    def calc_importance_weights(self, next_states, dones):
        with torch.no_grad():
            next_actions=(self.actor_target(next_obs, act_rew) + noise).clamp(-self.max_action, self.max_action)
            next_errs1, next_errs2 = \
                self._target_error_net(next_states, next_actions)

        # Terms inside the exponent of importance weights.
        if self.no_tau:
            x1 = -(1.0 - dones) * self._gamma * next_errs1
            x2 = -(1.0 - dones) * self._gamma * next_errs2
        else:
            x1 = -(1.0 - dones) * self._gamma * next_errs1 / (self._tau1 * self.tau_scale)
            x2 = -(1.0 - dones) * self._gamma * next_errs2 / (self._tau2 * self.tau_scale)


        # Calculate self-normalized importance weights.
        imp_ws1 = F.softmax(x1, dim=0)
        imp_ws2 = F.softmax(x2, dim=0)

        return imp_ws1, imp_ws2
    def train_TD3(
                self,
                replay_buffer=None,
                iterations=None,
                tasks_buffer = None,
                train_iter = 0,
                task_id = None,
                nums_snap_trains = 5):

        '''
            inputs:
                replay_buffer
                iterations episode_timesteps
            outputs:

        '''
        print('------train_TD3------')
        actor_loss_out = 0.0
        critic_loss_out = 0.0
        #todo 3.18qudiao if
        ### if there is no enough data in replay buffer, then reduce size of iteration to 20:
        #if replay_buffer.size_rb() < iterations or replay_buffer.size_rb() <  self.batch_size * iterations:
        #    temp = int( replay_buffer.size_rb()/ (self.batch_size) % iterations ) + 1
        #    if temp < iterations:
        #        iterations = temp

        for it in range(iterations):

            ########
            # Sample replay buffer
            ########
            x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample(self.batch_size)
            obs = torch.FloatTensor(x).to(self.device)
            next_obs = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            mask = torch.FloatTensor(1 - d).to(self.device)
            previous_action = torch.FloatTensor(pu).to(self.device)
            previous_reward = torch.FloatTensor(pr).to(self.device)
            previous_obs = torch.FloatTensor(px).to(self.device)

            # list of hist_actions and hist_rewards which are one time ahead of previous_ones
            # example:
            # previous_action = [t-3, t-2, t-1]
            # hist_actions    = [t-2, t-1, t]
            hist_actions = torch.FloatTensor(nu).to(self.device)
            hist_rewards = torch.FloatTensor(nr).to(self.device)
            hist_obs     = torch.FloatTensor(nx).to(self.device)

            # combine reward and action
            act_rew = [hist_actions, hist_rewards, hist_obs] # torch.cat([action, reward], dim = -1)
            pre_act_rew = [previous_action, previous_reward, previous_obs] #torch.cat([previous_action, previous_reward], dim = -1)

            ########
            # Select action according to policy and add clipped noise
            # mu'(s_t) = mu(s_t | \theta_t) + N (Eq.7 in https://arxiv.org/abs/1509.02971)
            # OR
            # Eq. 15 in TD3 paper:
            # e ~ clip(N(0, \sigma), -c, c)
            ########
            noise = (torch.randn_like(action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)
            #next_action = (self.actor_target(next_obs, act_rew) + noise).clamp(-self.max_action, self.max_action)
            next_action = (self.actor_target(next_obs, act_rew) + noise).clamp(-1.0, 1.0)
            #print('###mql583netx_action')
            ########
            #  Update critics
            #  1. Compute the target Q value
            #  2. Get current Q estimates
            #  3. Compute critic loss
            #  4. Optimize the critic
            ########

            # 1. y = r + \gamma * min{Q1, Q2} (s_next, next_action)
            # if done , then only use reward otherwise reward + (self.gamma * target_Q)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, act_rew)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (mask * self.gamma * target_Q).detach()

            # 2.  Get current Q estimates
            current_Q1, current_Q2 = self.critic(obs, action, pre_act_rew)

            # 3. Compute critic loss
            # even we picked min Q, we still need to backprob to both Qs
            #todo 3.18 daiding
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            critic_loss_out += critic_loss.item()

            # 4. Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ########
            # Delayed policy updates
            ########
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew).mean()
                actor_loss_out += actor_loss.item()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

        out = {}
        out['critic_loss'] = critic_loss_out/iterations
        out['actor_loss'] = self.policy_freq * actor_loss_out/iterations

        # keep a copy of models' params
        self.copy_model_params()
        return out, None

    def adapt(self,
            train_replay_buffer = None,
            train_tasks_buffer = None,
            eval_task_buffer = None,
            task_id = None,
            snap_iter_nums = 5,
            main_snap_iter_nums = 15,
            sampling_style = 'replay',
            sample_mult = 1
            ):
        '''
            inputs:
                replay_buffer
                iterations episode_timesteps
        '''
        #######
        # Reset optim at the beginning of the adaptation
        #######
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        print('------adapt------')
        #######
        # Adaptaion step:
        # learn a model to correct covariate shift
        #######
        out_single = None

        # train covariate shift correction model
        csc_model, csc_info = self.train_cs(task_id = task_id,
                                            snap_buffer = eval_task_buffer,
                                            train_tasks_buffer = train_tasks_buffer,
                                            adaptation_step = True)

        # train td3 for a single task
        out_single = self.do_training(replay_buffer = eval_task_buffer.get_buffer(task_id),
                                      iterations = snap_iter_nums,
                                      csc_model = None,
                                      apply_prox = False,
                                      current_batch_size = eval_task_buffer.size_rb(task_id))
        #self.copy_model_params()

        # keep a copy of model params for task task_id
        out_single['csc_info'] = csc_info
        out_single['snap_iter'] = snap_iter_nums

        # sampling_style is based on 'replay'
        # each train task has own buffer, so sample from each of them
        out = self.do_training(replay_buffer = train_replay_buffer,
                                   iterations = main_snap_iter_nums,
                                   csc_model = csc_model,
                                   apply_prox = True,
                                   current_batch_size = sample_mult * self.batch_size)

        return out, out_single

    def rollback(self):
        '''
            This function rollback everything to state before test-adaptation
        '''

        ####### ####### ####### Super Important ####### ####### #######
        # It is very important to make sure that we rollback everything to
        # Step 0
        ####### ####### ####### ####### ####### ####### ####### #######
        self.actor.load_state_dict(self.actor_copy.state_dict())
        self.actor_target.load_state_dict(self.actor_target_copy.state_dict())
        self.critic.load_state_dict(self.critic_copy.state_dict())
        self.critic_target.load_state_dict(self.critic_target_copy.state_dict())
        self.actor_optimizer.load_state_dict(self.actor_optimizer_copy.state_dict())
        self.critic_optimizer.load_state_dict(self.critic_optimizer_copy.state_dict())

    def save_model_states(self):

        ####### ####### ####### Super Important ####### ####### #######
        # Step 0: It is very important to make sure that we save model params before
        # do anything here
        ####### ####### ####### ####### ####### ####### ####### #######
        self.actor_copy = deepcopy(self.actor)
        self.actor_target_copy = deepcopy(self.actor_target)
        self.critic_copy = deepcopy(self.critic)
        self.critic_target_copy = deepcopy(self.critic_target)
        self.actor_optimizer_copy  = deepcopy(self.actor_optimizer)
        self.critic_optimizer_copy = deepcopy(self.critic_optimizer)

    def set_training_style(self):
        '''
            This function just selects style of training
        '''
        print('**** TD3 style is selected ****')
        self.training_func = self.train_TD3

    def train(self,
              replay_buffer = None,
              iterations = None,
              tasks_buffer = None,
              train_iter = 0,
              task_id = None,
              nums_snap_trains = 5):
        '''
         This starts type of desired training
        '''
        return self.training_func(  replay_buffer = replay_buffer,
                                    iterations = iterations,
                                    tasks_buffer = tasks_buffer,
                                    train_iter = train_iter,
                                    task_id = task_id,
                                    nums_snap_trains = nums_snap_trains
                                )

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.advantage_generator_net.state_dict(), filename + "_advantage_generator_net")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.advantage_generator_net.load_state_dict(torch.load(filename +"_advantage_generator_net"))
