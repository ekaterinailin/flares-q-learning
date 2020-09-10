"""
UTF-8, Python 3

------------------
FlareFairy
------------------

Ekaterina Ilin
2020, MIT License

Here we define reward functions for the FlareFairy class.
"""

import numpy as np
import pandas as pd

def get_reward_gauss(flare, recovered, w, w2, **kwargs):
    """Calculate a reward.
    
    Parameters:
    ------------
    flare : pandas Series
        should contain ampl_rec and dur of the
        flare you want to characterize
    recovered : [float, float]
        ampl_rec and dur of the recovered flare
    w : [float, float]
        weights for Gauss standard deviation for
        amplitude and duration, respectively
    w2 : [float, float]
        relative weights for Gauss amplitude for
        amplitude and duration, respectively
    kwargs : dict
        keyword arguments to pass to is_done
        
    Return:
    -------
    bool
    """
    
    # check if you are already done
    done = is_done(flare, recovered, **kwargs)
    
    # if nothing was recovered give minimum reward
    if (~np.isfinite(recovered)).any():
        return -1

    # but if there war a flare recovered
    elif np.isfinite(recovered).all():   
        
        # give maximum reward if done
        if done:
            return 1.
            
        # otherwise calculate reward based on 
        # the difference to real flare
        elif ~done:
            _ = zip([flare.ampl_rec, flare.dur], 
                    recovered, w, w2)
            # reward grows as a Gaussian bell curve, with maximum at the
            # flare location
            gauss_rewards = [w2 * np.exp( - (b - a)**2 / (2 * w**2)) for a, b, w, w2 in _]
            
            # normalize the reward to stay between -1 and 0
            return np.sum(gauss_rewards) / np.sum(w2) - 1. 
            


def is_done(flare, recovered, epsa=.05, epsd=.05):
    """If difference between recovered and real flare 
    amplitude and duration both fall below a given threshold
    the algorithm has reached its goal.
    
    Parameters:
    ------------
    flare : pandas Series
        should contain ampl_rec and dur of the
        flare you want to characterize
    recovered : [float, float]
        ampl_rec and dur of the recovered flare
    epsa : float < 1
        relative difference threshold between flare
        and synthetic event amplitude
    epsd : float < 1
        relative difference threshold between flare
        and synthetic event duration
        
    Return:
    -------
    bool
    """
    
    adone = np.abs((flare.ampl_rec - recovered[0]) / flare.ampl_rec) < epsa
    ddone = np.abs((flare.dur - recovered[1]) / flare.dur) < epsd
    
    return adone & ddone