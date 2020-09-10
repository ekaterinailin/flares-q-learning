import pytest

import numpy as np
import pandas as pd

from ..rewards import (is_done,
                       get_reward_gauss)

# ---------------------- TESTING get_reward_gauss() ----------------------

def test_get_reward_gauss():
    # define a minimum test flare
    flare = pd.Series({"ampl_rec":.1, "dur":.01})

    # define weights
    w=[.1,.005]
    w2=[1., 5.]

    # nailed it? - get full reward
    assert get_reward_gauss(flare, [.1,.01], w, w2) == 1.

    # somewhat off in one parameter reward is negative
    assert get_reward_gauss(flare, [.1,.005], w, w2) == pytest.approx(-0.3278, rel=1e-3)

    # reward grows if closer to the goal
    assert get_reward_gauss(flare, [.1,.005], w, w2) < get_reward_gauss(flare, [.1,.008], w, w2)
    assert get_reward_gauss(flare, [.14,.008], w, w2) < get_reward_gauss(flare, [.1,.008], w, w2)

    # get nearly smallest reward
    assert get_reward_gauss(flare, [1, .03], w, w2) == pytest.approx(- 1. + 1e-2, rel=1e-2)

    # nothing recovered - minimum reward
    assert get_reward_gauss(flare, [np.nan, .03], w, w2) == -1
    assert get_reward_gauss(flare, [.1, np.nan], w, w2) == -1

    # ill-defined flare throws error
    with pytest.raises(AttributeError):
        get_reward_gauss(pd.Series({"ampl_rec":.1, "bur":.01}), [.1, np.nan], w, w2)
        
# -------------------------------------------------------------------

# ---------------------- TESTING is_done() --------------------------

def test_is_done():
    # define a minimum test flare
    flare = pd.Series({"ampl_rec":.1, "dur":.01})

    # exact fit should test True
    assert is_done(flare, [.1,.01], epsa=.05, epsd=.05)

    # NaN value tests False
    assert ~is_done(flare, [.1,np.nan], epsa=.05, epsd=.05)

    # wrong input
    with pytest.raises(IndexError):
        is_done(flare, [np.nan,], epsa=.05, epsd=.05)

    # test big difference    
    assert ~is_done(flare, [0.5,.01], epsa=.05, epsd=.05)

    # 4 percent difference should be fine
    assert is_done(flare, [0.104,.0104], epsa=.05, epsd=.05) 

    # if the flare is ill-defined throw error
    with pytest.raises(AttributeError):
        is_done(pd.Series({"ampl_rec":.1, "bur":.01}), 
                [0.104,.0104], epsa=.05, epsd=.05) 

    # if NaN in flare event
    assert ~is_done(pd.Series({"ampl_rec":.1, "dur":np.nan}), 
                [0.104,.0104], epsa=.05, epsd=.05) 
    
# -------------------------------------------------------------------