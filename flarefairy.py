"""
UTF-8, Python 3

------------------
FlareFairy
------------------

Ekaterina Ilin
<add your name>
<add your name>
<add your name>
2020, MIT License

This is the main Q-learning class FlareFairy.

"""

import numpy as np
import pandas as pd


class FlareFairy:
    """NEEDS SOME DOCS TO EXPLAIN THE ATTRIBUTES
    
    Attributes:
    -----------
    flc : FlareLightCurve object
    flare : pandas.Series object
        contains values for "ampl_rec", "dur", can be a row
        from flc.flares
    amax : float
        factor that determines the minium and 
        maximum injected flare amplitudes on the grid
    dmax : float
        factor that determines the minium and 
        maximum injected flare fwhm on the grid
    DISCRETE_OS_SIZE : [int, int]
        dimensions of q table
    LEARNING_RATE: float < 1
    DISCOUT: float < 1
    """
    
    def __init__(self, flc, flare, amax=5., dmax=5., DISCRETE_OS_SIZE=[20, 20],
                 LEARNING_RATE=.5, DISCOUNT=.75):
        
        self.DISCRETE_OS_SIZE = DISCRETE_OS_SIZE 
        self.LEARNING_RATE = LEARNING_RATE
        self.DISCOUNT = DISCOUNT
        
        
        self.flc = flc
        self.flare = flare
        self.amax = flare.ampl_rec * amax # maximum flare amplitude on the grid
        self.dmax = flare.dur * dmax / 6 # maximum flare fwhm on the grid
        self.amin = flare.ampl_rec / amax # minimum flare amplitude on the grid 
        self.dmin = flare.dur / dmax / 6 # minimum flare fwhm on the grid
        
        # set up window size for q table
        self.discrete_os_win_size = ((np.array([self.amax,self.dmax]) -
                                      np.array([self.amin,self.dmin])) /
                                     self.DISCRETE_OS_SIZE )
        # get step size for amplitude
        self.astep = self.discrete_os_win_size[0]
        # get step size for amplitude
        self.dstep = self.discrete_os_win_size[1]
        # define steps 0-3
        # each step is [axis, step size and direction, upper/lower bound, function]
        self.steps = {0:[0,self.astep,self.amax*.999, min],
                      1:[0,-self.astep,self.amin, max],
                      2:[1,self.dstep,self.dmax*.999, min],
                      3:[1,-self.dstep, self.dmin,max]}
        # in the beginning the fairy assumes that the injected flare is roughly the 
        # same as the detected one
        self.state = [flare.ampl_rec, flare.dur / 6]
        
        # we have not yet recovered anything 
        self.recovered = [np.nan, np.nan]
        
        # in the end we want the recovered flare to match the observed one
        self.goal = [flare.ampl_rec, flare.dur]

        # optionally set weights in the reward function
        self.weights = [1.,1.]
        
        # define a minimum reward
        self.min_reward = -1.
        
        # save injections to table
        self.injections = pd.DataFrame()
        
        # get initial reward and check if we are already done
        self.get_reward()
        self.is_done()
        
        # define q table
        self.q_table = np.random.uniform(low=self.min_reward, high=0, size=(self.DISCRETE_OS_SIZE + [len(self.steps)]))
       
    def reset(self):
        self.state = [np.random.uniform(low=self.amin, high=self.amax),
                      np.random.uniform(low=self.dmin, high=self.dmax)]
        self.is_done()
        self.get_reward()

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
            self.recovered = [np.nan, np.nan]
            
        
    def get_reward(self):
        if np.isnan(self.recovered).any():
            self.reward = -.1#self.min_reward
        else:
            _ = zip(self.goal, self.recovered, self.weights)
            self.reward = - np.sum([((a - b) * w)**2 for a, b, w in _])
            print("reward ", self.reward)
        
    def is_done(self, epsa=.07, epsd=.007):
        adone = np.abs(self.goal[0] - self.recovered[0]) < epsa
        ddone = np.abs(self.goal[1] - self.recovered[1]) < epsd
        self.done = adone & ddone

    def action(self, x):
        self.move(x)
        res = self.evaluate()
        self.get_reward()
        self.is_done
        return self.get_discrete_state()
    
    def run_episode(self, nsteps=30, epsilon=0):
        i = 0
        while not self.done:

            currentstate = self.get_discrete_state()
            
            if np.random.random() > epsilon:
                # Get action from Q table
                x = np.argmax(self.q_table[currentstate])
            else:
                # Get random action
                x = np.random.randint(0, len(self.steps))
           
            newstate = self.action(x)

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(self.q_table[newstate])

            # Current Q value (for current state and performed action)
            current_q = self.q_table[currentstate + (x,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (self.reward + self.DISCOUNT * max_future_q)
            print("currentQ, newQ ", current_q, new_q)
            # Update Q table with new Q value
            self.q_table[currentstate + (x,)] = new_q

            i += 1
            if i > nsteps:
                self.done = True
        print(f"Finished episode after {i} steps.")
        self.is_done()
        if self.done:
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
        return end_df    
