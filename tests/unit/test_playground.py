import pytest

from gym import Env
from learnrl import Playground, Agent


def test_single_agent():
    env = Env()
    agents = Agent()
    pg = Playground(env, agents)
    if pg.agents != [agents]:
        raise ValueError(
            "A single agent should be transformed in a list containing itself"
        )

def test_agent_order():
    env = Env()
    n_agents = 5
    agents = [Agent() for _ in range(n_agents)]

    # Default order
    pg = Playground(env, agents)
    if pg.agents_order != list(range(n_agents)):
        raise ValueError(
            f"Default agents_order shoud be {list(range(n_agents))}"
            f"but was {pg.agents_order}"
        )

    # Custom order at init
    custom_order = [4, 3, 1, 2, 0]
    pg = Playground(env, agents, agents_order=custom_order)
    if pg.agents_order != custom_order:
        raise ValueError(
            f"Custom agents_order shoud be {custom_order}"
            f"but was {pg.agents_order}"
        )

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
    pg.set_agents_order(new_custom_order)

    if pg.agents_order != new_custom_order:
        raise ValueError(
            f"Custom agents_order shoud be {new_custom_order}"
            f"but was {pg.agents_order}"
        )


