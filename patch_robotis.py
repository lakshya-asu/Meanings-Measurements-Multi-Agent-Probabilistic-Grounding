import re

with open("src/real_world/robotis_hydra_agent.py", "r") as f:
    code = f.read()

# 1. Rename class
code = code.replace("class RobotHydraAgent:", "class RobotisHydraAgent:")

# 2. __init__ args
code = code.replace("hydra_pipeline,", "")
code = code.replace("self.hydra_pipeline = hydra_pipeline", "")
code = code.replace('self.hydra_update_freq = parameters["hydra_update_freq"]', "")

# 3. Remove _start_threads call
code = code.replace("self._start_threads()", "")

# 4. Remove hydra_step from update
code = code.replace("self.hydra_step(obs)", "self.update_frontiers()")

# 5. Remove hydra mesh from rerun
code = code.replace("self.robot._rerun.update_hydra_mesh(self.hydra_pipeline)", "")

with open("src/real_world/robotis_hydra_agent.py", "w") as f:
    f.write(code)
print("Patched!")
