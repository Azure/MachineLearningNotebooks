import time
import glob
import pathlib

from malmo import MalmoPython, malmoutils
from malmo.launch_minecraft_in_background import launch_minecraft_in_background


class MalmoVideoRecorder:
    DEFAULT_RECORDINGS_DIR = './logs/videos'

    def __init__(self):
        self.agent_host_bot = None
        self.agent_host_camera = None
        self.client_pool = None
        self.is_malmo_initialized = False

    def init_malmo(self, recordings_directory=DEFAULT_RECORDINGS_DIR):
        if self.is_malmo_initialized:
            return

        launch_minecraft_in_background(
            '/app/MalmoPlatform/Minecraft',
            ports=[10000, 10001])

        # Set up two agent hosts
        self.agent_host_bot = MalmoPython.AgentHost()
        self.agent_host_camera = MalmoPython.AgentHost()

        # Create list of Minecraft clients to attach to. The agents must
        # have been launched before calling record_malmo_video using
        # init_malmo()
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool.add(MalmoPython.ClientInfo('127.0.0.1', 10000))
        self.client_pool.add(MalmoPython.ClientInfo('127.0.0.1', 10001))

        # Use bot's agenthost to hold the command-line options
        malmoutils.parse_command_line(
            self.agent_host_bot,
            ['--record_video', '--recording_dir', recordings_directory])

        self.is_malmo_initialized = True

    def _start_mission(self, agent_host, mission, recording_spec, role):
        used_attempts = 0
        max_attempts = 5

        while True:
            try:
                agent_host.startMission(
                    mission,
                    self.client_pool,
                    recording_spec,
                    role,
                    '')
                break
            except MalmoPython.MissionException as e:
                errorCode = e.details.errorCode
                if errorCode == (MalmoPython.MissionErrorCode
                                 .MISSION_SERVER_WARMING_UP):
                    time.sleep(2)
                elif errorCode == (MalmoPython.MissionErrorCode
                                   .MISSION_INSUFFICIENT_CLIENTS_AVAILABLE):
                    print('Not enough Minecraft instances running.')
                    used_attempts += 1
                    if used_attempts < max_attempts:
                        print('Will wait in case they are starting up.')
                        time.sleep(300)
                elif errorCode == (MalmoPython.MissionErrorCode
                                   .MISSION_SERVER_NOT_FOUND):
                    print('Server not found.')
                    used_attempts += 1
                    if used_attempts < max_attempts:
                        print('Will wait and retry.')
                        time.sleep(2)
                else:
                    used_attempts = max_attempts
                if used_attempts >= max_attempts:
                    raise e

    def _wait_for_start(self, agent_hosts):
        start_flags = [False for a in agent_hosts]
        start_time = time.time()
        time_out = 120

        while not all(start_flags) and time.time() - start_time < time_out:
            states = [a.peekWorldState() for a in agent_hosts]
            start_flags = [w.has_mission_begun for w in states]
            errors = [e for w in states for e in w.errors]

            if len(errors) > 0:
                print("Errors waiting for mission start:")
                for e in errors:
                    print(e.text)
                raise Exception("Encountered errors while starting mission.")
        if time.time() - start_time >= time_out:
            raise Exception("Timed out while waiting for mission to start.")

    def _get_xml(self, xml_file, seed):
        with open(xml_file, 'r') as mission_file:
            return mission_file.read().format(SEED_PLACEHOLDER=seed)

    def _is_mission_running(self):
        return self.agent_host_bot.peekWorldState().is_mission_running or \
            self.agent_host_camera.peekWorldState().is_mission_running

    def record_malmo_video(self, instructions, xml_file, seed):
        '''
        Replays a set of instructions through Malmo using two players.  The
        first player will navigate the specified mission based on the given
        instructions.  The second player observes the first player's moves,
        which is captured in a video.
        '''

        if not self.is_malmo_initialized:
            raise Exception('Malmo not initialized. Call init_malmo() first.')

        # Set up the mission
        my_mission = MalmoPython.MissionSpec(
            self._get_xml(xml_file, seed),
            True)

        bot_recording_spec = MalmoPython.MissionRecordSpec()
        camera_recording_spec = MalmoPython.MissionRecordSpec()

        recordingsDirectory = \
            malmoutils.get_recordings_directory(self.agent_host_bot)
        if recordingsDirectory:
            camera_recording_spec.setDestination(
                recordingsDirectory + "//rollout_" + str(seed) + ".tgz")
            camera_recording_spec.recordMP4(
                MalmoPython.FrameType.VIDEO,
                36,
                2000000,
                False)

        # Start the agents
        self._start_mission(
            self.agent_host_bot,
            my_mission,
            bot_recording_spec,
            0)
        self._start_mission(
            self.agent_host_camera,
            my_mission,
            camera_recording_spec,
            1)
        self._wait_for_start([self.agent_host_camera, self.agent_host_bot])

        # Teleport the camera agent to the required position
        self.agent_host_camera.sendCommand('tp -29 72 -6.7')
        instruction_index = 0

        while self._is_mission_running():

            command = instructions[instruction_index]
            instruction_index += 1

            self.agent_host_bot.sendCommand(command)

            # Pause for half a second - change this for faster/slower videos
            time.sleep(0.5)

            if instruction_index == len(instructions):
                self.agent_host_bot.sendCommand("jump 1")
                time.sleep(2)

                self.agent_host_bot.sendCommand("quit")

                # Wait a little for Malmo to reset before the
                # next mission is started
                time.sleep(2)
                print("Video recorded.")
