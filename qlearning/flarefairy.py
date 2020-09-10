"""
UTF-8, Python 3

------------------
FlareFairy
------------------

Ekaterina Ilin
2020, MIT License

This is the main Q-learning class FlareFairy.

"""

import numpy as np
import pandas as pd

import time

from .rewards import is_done
from .rewards import get_reward_gauss

class FlareFairy:
    """NEEDS SOME DOCS TO EXPLAIN THE ATTRIBUTES
    
    Attributes:
    -----------
    flc : FlareLightCurve object
    flare : pandas.Series object
        contains values for "ampl_rec", "dur", can be a row
        from flc.flares
    """
    
    def __init__(self, flc, flare, amax=5., dmax=5., DISCRETE_OS_SIZE=[20, 20],
                 LEARNING_RATE=.9, DISCOUNT=.75, 
                 thresh_func=is_done, 
                 thresh_func_params={"epsa":.05, "epsd":.05},
                 reward_func=get_reward_gauss,):
        
        self.DISCRETE_OS_SIZE = DISCRETE_OS_SIZE 
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT = DISCOUNT
        
        
        self.flc = flc
        self.flare = flare
        
        # Don't run the Fairy on ill-defined flares
        if (self.flare[["ampl_rec","dur"]].isnull()).any():
            raise ValueError("Flare event has non-finite elements in the critical"
                             " attributes ampl_rec and/or dur.")
        
        self.amax = flare.ampl_rec * amax
        self.dmax = flare.dur * dmax / 6
        self.amin = flare.ampl_rec / amax
        self.dmin = flare.dur / dmax /6
        
        self.discrete_os_win_size = ((np.array([self.amax,self.dmax]) -
                                      np.array([self.amin,self.dmin])) /
                                     self.DISCRETE_OS_SIZE )
        self.astep = self.discrete_os_win_size[0]
        self.dstep = self.discrete_os_win_size[1]
        self.steps = {0:[0,self.astep,self.amax*.999, min],
                      1:[0,-self.astep,self.amin, max],
                      2:[1,self.dstep,self.dmax*.999, min],
                      3:[1,-self.dstep, self.dmin,max]}
        self.state = [flare.ampl_rec, flare.dur / 6]
        self.recovered = [np.nan, np.nan]
        self.goal = [flare.ampl_rec, flare.dur]
        self.min_reward = -1.
       
        self.injections = pd.DataFrame()
        
        self.q_table = np.random.uniform(low=self.min_reward, high=0, size=(self.DISCRETE_OS_SIZE + [len(self.steps)]))
        
        # define reward function
        def get_reward(recovered):
            return reward_func(self.flare, recovered, 
                               [self.astep*5., self.dstep*10],
                               [1., self.flare.ampl_rec / self.flare.dur / 2],
                               **thresh_func_params)
        
        self.get_reward = get_reward
        
        self.reward = self.get_reward(self.recovered)
        
        # define finish threshold function
        def check_if_done(recovered):
            return thresh_func(self.flare, recovered, **thresh_func_params)
        
        self.check_if_done = check_if_done
        
        self.done = self.check_if_done(self.recovered)
       
    def reset(self):
        self.state = [np.random.uniform(low=self.amin, high=self.amax),
                      np.random.uniform(low=self.dmin, high=self.dmax)]
        self.done = self.check_if_done(self.recovered)
        self.reward = self.get_reward(self.recovered)

    def get_discrete_state(self):
        discrete_state = (np.array(self.state) - np.array([self.amin,self.dmin])) / self.discrete_os_win_size
        return tuple(discrete_state.astype(np.int))

    def move(self, x):
        idx, val, bound, func = self.steps[x]
        self.state[idx] = func(self.state[idx] + val, bound)
        
    def evaluate(self):
        try:
            flc, fake_lc = self.flc.sample_flare_recovery(inject_before_detrending=False,# mode="savgol",
                                              iterations=1, fakefreq=1e-5, #show_progress=False,
                                              ampl=[self.state[0]-self.astep/4,self.state[0]+self.astep/4],
                                              dur=[self.state[1]-self.dstep/4,self.state[1]+self.dstep/4])
            
            res = flc.fake_flares.iloc[0]
        
            self.injections = self.injections.append(res,ignore_index=True)
            self.recovered = [res.ampl_rec, res.dur]
        
            self.full_recovered = res
        except Exception as e: #somethin goes wrong with injection recovery
            print("EXCEPTION: ", e)
            self.recovered = [-1.,-1.]#[np.nan, np.nan]
        
            
    

    def action(self, x):
        
        self.move(x)
        
        res = self.evaluate()
        
        self.reward = self.get_reward(self.recovered)
        
        self.done = self.check_if_done(self.recovered)
        
        return self.get_discrete_state()
    
    def run_episode(self, nsteps=100, epsilon=0):
        
        i = 0
        currentstate = self.get_discrete_state() 
        
        if np.random.random() > epsilon:
            # Get action from Q table
            x = np.argmax(self.q_table[currentstate])
        else:
            # Get random action
            x = np.random.randint(0, len(self.steps))
        while not self.done:

            newstate = self.action(x)

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(self.q_table[newstate])

            # Current Q value (for current state and performed action)
            current_q = self.q_table[currentstate + (x,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (self.reward + self.DISCOUNT * max_future_q)
          
            # Update Q table with new Q value
            self.q_table[currentstate + (x,)] = new_q

            i += 1
            if i > nsteps:
                self.done = True
        print(f"Finished episode after {i} steps.")
        
        if self.check_if_done(self.recovered):
            # Update Q table with new Q value
            self.q_table[currentstate + (x,)] = 0
            
    def run_n_episodes(self, n):
        epsilon = 1  # not a constant, qoing to be decayed
        START_EPSILON_DECAYING = 1
        END_EPSILON_DECAYING = n//2
        epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
        end_df = pd.DataFrame()
        for i in range(n):
            self.reset()
            self.run_episode(epsilon=epsilon)
            end_df = end_df.append(self.full_recovered, ignore_index=True)
            if END_EPSILON_DECAYING >= i >= START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value
            time.sleep(5)
        return end_df    