CONFIG_MAPPING = {}

try:
    from experiments.task1_pick_banana.config import TrainConfig as PickBananaTrainConfig
    CONFIG_MAPPING["task1_pick_banana"] = PickBananaTrainConfig
except Exception:
    pass

try:
    from experiments.task1_lift_cube_sim.config import TrainConfig as LiftCubeSimTrainConfig
    CONFIG_MAPPING["task1_lift_cube_sim"] = LiftCubeSimTrainConfig
except Exception as _e:
    pass