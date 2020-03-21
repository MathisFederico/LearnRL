import pytest
import numpy as np

from copy import deepcopy
from learnrl.core import Memory


class AgentMemoryError(Exception):
    pass

"""Test the behavior of the general memory for agents"""
@pytest.fixture
def memory():
    """Create a memory instance"""
    return Memory()

def test_forget(memory):
    """Test that we do reset memory"""
    initial_datas = {key:None for key in memory.MEMORY_KEYS}

    memory.remember(0, 0, 0, False, 0, info={'parameter':0})
    if memory.datas == initial_datas:
        raise AgentMemoryError('We are suppose to remember a dummy experience')
    
    memory.forget()
    for key in initial_datas:
        if memory.datas[key] is not None:
            raise AgentMemoryError(f'We are suppose to have forgotten the dummy experience \
                    \nBut in key {key}\
                    \nWe still remember {memory.datas[key]}\
                    \nInstead of {initial_datas[key]}')

def test_remember(memory):
    """
    As a reminder, we ought to remember at least those MEMORY_KEYS : (state, action, reward, done, next_state, info)
    For state, action, next_state :
        Test that we can remember int, floats, lists, or numpy.ndarrays as numpy.ndarrays
    For reward :
        Test that we can remember int, floats as numpy.ndarrays
    For done :
        Test that we can remember bool as numpy.ndarrays
    For info :
        Test that we can remember dict as a 2D numpy.ndarrays
    Test that for a same key in MEMORY_KEYS, datas have consitant shapes to create a numpy.ndarrays (except info)
    """

    # Test int and float behavior
    memory.remember(state=0, action=1.0, reward=2, done=False, next_state=3, info={'param':4})
    memory.remember(state=5, action=6.0, reward=7, done=True, next_state=8, info={'param':9})
    expected = {'state':np.array([0, 5]), 'action':np.array([1.0, 6.0]), 'reward':np.array([2, 7]),
                'done':np.array([0, 1]), 'next_state':np.array([3, 8]), 'info':np.array([{'param':4}, {'param':9}])}
    for key in memory.MEMORY_KEYS:
        if not np.all(memory.datas[key]==expected[key]):
            raise AgentMemoryError(f'Couldn\'t remember the int&float dummy exemple !\
                        \nKey : {key}\nGot \n{memory.datas[key]}\ninstead of\n{expected[key]}')
    memory.forget()

    # Test list behavior
    memory.remember(state=[0, 1], action=[2, 3], reward=4, done=False, next_state=[5, 6], info={'param':7})
    memory.remember(state=[8, 9], action=[10, 11], reward=12, done=True, next_state=[13, 14], info={'param':15})
    expected = {'state':np.array([[0, 1], [8, 9]]), 'action':np.array([[2, 3], [10, 11]]), 'reward':np.array([4, 12]),
                'done':np.array([0, 1]), 'next_state':np.array([[5, 6], [13, 14]]), 'info':np.array([{'param':7}, {'param':15}])}
    for key in memory.MEMORY_KEYS:
        if not np.all(memory.datas[key]==expected[key]):
            raise AgentMemoryError(f'Couldn\'t remember the list dummy exemple !\
                        \n Key : {key}, Got {memory.datas[key]} instead of {expected[key]}')
    memory.forget()

    # Test shape consistensy forced by numpy.ndarrays
    with pytest.raises(ValueError):
        memory.remember(state=0, action=1.0, reward=2, done=False, next_state=3, info={'param':4})
        memory.remember(state=[0, 1], action=[2, 3], reward=4, done=False, next_state=[5, 6], info={'param':7})

