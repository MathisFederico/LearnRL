import pytest

from gym import Env
from learnrl import Playground, Agent


def test_single_agent():
    env = Env()
    agents = Agent()
    pg = Playground(env, agents)
    assert pg.agents == [agents]

def test_agent_order():
    env = Env()
    n_agents = 5
    agents = [Agent() for _ in range(n_agents)]

    # Default order
    pg = Playground(env, agents)
    assert pg.agents_order == list(range(n_agents))

    # Custom order at init
    custom_order = [4, 3, 1, 2, 0]
    pg = Playground(env, agents, agents_order=custom_order)
    assert pg.agents_order == custom_order

    # Not enough indexes in custom order
    with pytest.raises(ValueError, match=r"Not every agents*"):
        pg = Playground(env, agents, agents_order=[4, 3, 1, 2])

    # Missing indexes in custom order
    with pytest.raises(ValueError, match=r".*not taking every index*"):
        pg = Playground(env, agents, agents_order=[4, 6, 1, 2, 0])
    
    # Custom order after init
    custom_order = [4, 3, 1, 2, 0]
    pg = Playground(env, agents, agents_order=custom_order)

    new_custom_order = [3, 4, 1, 2, 0]
    pg.set_agent_order(new_custom_order)

    assert pg.agents_order == new_custom_order


