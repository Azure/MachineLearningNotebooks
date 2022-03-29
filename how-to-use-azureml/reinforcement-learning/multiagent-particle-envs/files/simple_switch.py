import numpy as np
import random

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class SwitchWorld(World):
    """ Extended World with hills and switches """
    def __init__(self, hills, switches):
        super().__init__()
        # add hills and switches
        self.hills = hills
        self.switches = switches
        self.landmarks.extend(self.hills)
        self.landmarks.extend(self.switches)

    def step(self):

        super().step()

        # if all hills are activated, reset the switches and hills
        if all([hill.active for hill in self.hills]):
            self.reset_hills()
            self.reset_switches()
        else:
            # Update switches
            for switch in self.switches:
                switch.step(self)
            # Update hills
            for hill in self.hills:
                hill.step(self)

    def reset_hills(self):
        possible_hill_positions = [np.array([-0.8, 0]), np.array([0, 0.8]), np.array([0.8, 0]), np.array([0, -0.8])]
        hill_positions = random.sample(possible_hill_positions, k=len(self.hills))
        for i, hill in enumerate(self.hills):
            hill.state.p_pos = hill_positions[i]
            hill.deactivate()

    def reset_switches(self):
        possible_switch_positions = [
            np.array([-0.8, -0.8]),
            np.array([-0.8, 0.8]),
            np.array([0.8, -0.8]),
            np.array([0.8, 0.8])]
        switch_positions = random.sample(possible_switch_positions, k=len(self.switches))
        for i, switch in enumerate(self.switches):
            switch.state.p_pos = switch_positions[i]
            switch.deactivate()


class Scenario(BaseScenario):
    def make_world(self):

        # main configurations
        num_agents = 2
        num_hills = 2
        num_switches = 1
        self.max_episode_length = 100

        # create hills (on edges)
        possible_hill_positions = [np.array([-0.8, 0]), np.array([0, 0.8]), np.array([0.8, 0]), np.array([0, -0.8])]
        hill_positions = random.sample(possible_hill_positions, k=num_hills)
        hills = [Hill(hill_positions[i]) for i in range(num_hills)]
        # create switches (in corners)
        possible_switch_positions = [
            np.array([-0.8, -0.8]),
            np.array([-0.8, 0.8]),
            np.array([0.8, -0.8]),
            np.array([0.8, 0.8])]
        switch_positions = random.sample(possible_switch_positions, k=num_switches)
        switches = [Switch(switch_positions[i]) for i in range(num_switches)]

        # make world and set basic properties
        world = SwitchWorld(hills, switches)
        world.dim_c = 2
        world.collaborative = True

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
            agent.accel = 5.0
            agent.max_speed = 5.0
            if i == 0:
                agent.color = np.array([0.35, 0.35, 0.85])
            else:
                agent.color = np.array([0.35, 0.85, 0.85])

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.array([random.uniform(-1, +1) for _ in range(world.dim_p)])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # set hills randomly
        world.reset_hills()
        # set switches randomly
        world.reset_switches()

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on number of landmarks activated
        rew = 0
        if all([h.active for h in world.hills]):
            rew += 100
        else:
            # give bonus each time a hill is activated
            for hill in world.hills:
                if hill.activated_just_now:
                    rew += 50
        # penalise timesteps where nothing is happening
        if rew == 0:
            rew -= 0.1
        # add collision penalty
        if agent.collide:
            for a in world.agents:
                # note: this also counts collision with "itself", so gives -1 at every timestep
                # would be good to tune the reward function and use (not a == agent) here
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)


class Hill(Landmark):
    """
    A hill that can be captured by an agent.
    To be captured, a team must occupy a hill for a fixed amount of time.
    """

    def __init__(self,
                 pos=None,
                 size=0.08,
                 capture_time=2
                 ):

        # Initialize Landmark super class
        super().__init__()
        self.movable = False
        self.collide = False
        self.state.p_pos = pos
        self.size = size

        # Set static configurations
        self.capture_time = capture_time

        # Initialize all hills to be inactive
        self.active = False
        self.color = np.array([0.5, 0.5, 0.5])
        self.capture_timer = 0

        self.activated_just_now = False

    def activate(self):
        self.active = True
        self.color = np.array([0.1, 0.1, 0.9])

    def deactivate(self):
        self.active = False
        self.color = np.array([0.5, 0.5, 0.5])

    def _is_occupied(self, agents):
        # a hill is occupied if an agent stands on it
        for agent in agents:
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - self.state.p_pos)))
            if dist < agent.size + self.size:
                return True
        return False

    def step(self, world):

        self.activated_just_now = False

        # If hill isn't activated yet, check if an agent activates it
        # if (not self.active) and (world.switch.is_active()):
        if (not self.active):

            # Check if an agent is on the hill and all switches are active
            if (self._is_occupied(world.agents)) and all([switch.active for switch in world.switches]):
                self.capture_timer += 1

                # activate hill (this is irreversible)
                if self.capture_timer > self.capture_time:
                    self.activate()
                    self.activated_just_now = True

            # Reset capture timer if hill is not occupied
            else:
                self.capture_timer = 0


class Switch(Landmark):
    """
    A switch that can be activated by an agent.
    The agent has to stay on the switch for it to be active.
    """

    def __init__(self,
                 pos=None,
                 size=0.03,
                 ):

        # Initialize Landmark super class
        super().__init__()
        self.movable = False
        self.collide = False
        self.state.p_pos = pos
        self.size = size

        # Initialize all hills to be inactive
        self.active = False
        self.color = np.array([0.8, 0.05, 0.3])
        self.capture_timer = 0

    def activate(self):
        self.active = True
        self.color = np.array([0.1, 0.9, 0.4])

    def deactivate(self):
        self.active = False
        self.color = np.array([0.8, 0.05, 0.3])

    def _is_occupied(self, agents):
        # a switch is active if an agent stands on it
        for agent in agents:
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - self.state.p_pos)))
            if dist < agent.size + self.size:
                return True
        return False

    def step(self, world):
        # check if an agent is on the switch and activate/deactive accordingly
        if self._is_occupied(world.agents):
            self.activate()
        else:
            self.deactivate()


class SwitchExpertPolicy():
    """
    Hand-coded expert policy for the simple switch environment.
    Types of possible experts:
    - always go to the switch
    - always go to the hills
    """
    def __init__(self, dim_c, agent, world, expert_type=None, discrete_action_input=True):

        self.dim_c = dim_c
        self.discrete_action_input = discrete_action_input
        # the agent we control and world we're in
        self.agent = agent
        self.world = world

        if expert_type is None:
            self.expert_type = random.choice(['switch', 'hill'])
        else:
            self.expert_type = expert_type
        if self.expert_type == 'switch':
            self.target_switch = self.select_inital_target_switch()
        elif self.expert_type == 'hill':
            self.target_hill = self.select_inital_target_hill()
        else:
            raise NotImplementedError

        self.step_count = 0

    def select_inital_target_switch(self):
        return random.choice(self.world.switches)

    def select_inital_target_hill(self):
        return random.choice(self.world.hills)

    def action(self):

        # select a target!
        if self.expert_type == 'switch':
            # if agent is not already on a switch, choose target switch
            if not any([switch._is_occupied([self.agent]) for switch in self.world.switches]):
                # select a target switch if there's an inactive one
                inactive_switches = [switch for switch in self.world.switches if not switch.active]
                if len(inactive_switches) > 0 and (self.target_switch not in inactive_switches):
                    self.target_switch = random.choice(inactive_switches)
            target = self.target_switch.state.p_pos
        elif self.expert_type == 'hill':
            # select a target hill if we haven't done so yet, or the current target switch is inactive
            inactive_hills = [hill for hill in self.world.hills if not hill.active]
            if len(inactive_hills) > 0 and (self.target_hill not in inactive_hills):
                self.target_hill = random.choice(inactive_hills)
            target = self.target_hill.state.p_pos

        self.step_count += 1

        impulse = np.clip(target - self.agent.state.p_pos, -self.agent.u_range, self.agent.u_range)

        if self.discrete_action_input:
            u_idx = np.argmax(np.abs(impulse))
            if u_idx == 0 and impulse[u_idx] < 0:
                u = 1
            elif u_idx == 0 and impulse[u_idx] > 0:
                u = 2
            elif u_idx == 1 and impulse[u_idx] < 0:
                u = 3
            elif u_idx == 1 and impulse[u_idx] > 0:
                u = 4
            else:
                u = 0
        else:
            u = np.zeros(5)
            if (impulse[0] == impulse[1] == 0) \
                or (self.step_count < self.burn_in) \
                    or (self.burn_step != 0 and self.step_count % self.burn_step != 0):
                u[0] = 0.1
            else:
                pass
                # u: noop (?), right, left, down, up
                if impulse[0] > 0:  # x-direction (- left/right + )
                    u[1] = impulse[0]  # right
                elif impulse[0] < 0:
                    u[2] = -impulse[0]
                if impulse[1] > 0:  # y-direction (- up/down + )
                    u[3] = impulse[1]
                elif impulse[1] < 0:
                    u[4] = -impulse[1]

        return u
