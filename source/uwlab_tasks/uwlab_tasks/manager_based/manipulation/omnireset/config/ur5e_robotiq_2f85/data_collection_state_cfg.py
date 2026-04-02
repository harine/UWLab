from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from .rl_state_cfg import Ur5eRobotiq2f85RlStateCfg
from .actions import Ur5eRobotiq2f85RelativeOSCEvalAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from .rl_state_cfg import FinetuneEvalEventCfg, RlStateSceneCfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from ... import mdp as task_mdp

@configclass
class DataCollectionStateTerminationsCfg:

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    early_success = DoneTerm(
        func=task_mdp.early_success_termination, params={"num_consecutive_successes": 5, "min_episode_length": 10}
    )

    success = DoneTerm(
        func=task_mdp.consecutive_success_state_with_min_length,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )

@configclass
class StateObservationsCfg:
    @configclass
    class StatePolicyCfg(ObsGroup):
        """Observations for policy group (with processed images for evaluation)."""

        last_gripper_action = ObsTerm(
            func=task_mdp.last_action,
            params={
                "action_name": "gripper",
            },
        )

        last_arm_action = ObsTerm(
            func=task_mdp.last_action,
            params={
                "action_name": "arm",
            },
        )

        arm_joint_pos = ObsTerm(
            func=task_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"]),
            },
        )

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        # Privileged observations
        joint_vel = ObsTerm(func=task_mdp.joint_vel)

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class StateDataCollectionCfg(ObsGroup):
        """Observations for data collection group (with unprocessed images for saving)."""

        last_gripper_action = ObsTerm(
            func=task_mdp.last_action,
            params={
                "action_name": "gripper",
            },
        )

        last_arm_action = ObsTerm(
            func=task_mdp.last_action,
            params={
                "action_name": "arm",
            },
        )

        arm_joint_pos = ObsTerm(
            func=task_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"]),
            },
        )

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        # Additional observations
        binary_contact = ObsTerm(
            func=task_mdp.binary_force_contact,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "wrist_3_link",
                "force_threshold": 25.0,
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        joint_vel = ObsTerm(func=task_mdp.joint_vel)

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False
    

    # observation groups
    data_collection: StateDataCollectionCfg = StateDataCollectionCfg()
    policy: StatePolicyCfg = StatePolicyCfg()

@configclass
class EvalStateObservationsCfg(StateObservationsCfg):
    @configclass
    class CameraObservationsCfg(ObsGroup):
        front_rgb = ObsTerm(
                func=task_mdp.process_image,
                params={
                    "sensor_cfg": SceneEntityCfg("front_camera"),
                    "data_type": "rgb",
                    "process_image": True,
                    "output_size": (224, 224),
                },
            )

        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224),
            },
        )

        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224),
            },
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False
    camera: CameraObservationsCfg = CameraObservationsCfg()

@configclass 
class StateCommandsCfg:
    """Command specifications for the MDP."""

    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )

@configclass
class DataCollectionStateEventCfg(FinetuneEvalEventCfg):
    """Data collection events: override reset to sample from all 4 distributions."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class StateEvalEventCfg(FinetuneEvalEventCfg):
    """State evaluation events: override reset to sample from all 4 distributions."""
    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )

@configclass
class EvalStateSceneCfg(RlStateSceneCfg):
    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
        update_period=0,
        height=240,
        width=320,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.1679045, 0.4486344),
            rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
        update_period=0,
        height=240,
        width=320,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8323904, 0.5877843, 0.2805111),
            rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.10),
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=240,
        width=320,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.55),
    )

@configclass
class Ur5eRobotiq2f85StateRelCartesianOSCEvalCfg(Ur5eRobotiq2f85RlStateCfg):
    """State base config: fixed sysid + State scene/obs/terminations/render."""

    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()
    observations: StateObservationsCfg = StateObservationsCfg()
    terminations: DataCollectionStateTerminationsCfg = DataCollectionStateTerminationsCfg()
    commands: StateCommandsCfg = StateCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 32.0

@configclass
class Ur5eRobotiq2f85DataCollectionStateRelCartesianOSCCfg(Ur5eRobotiq2f85StateRelCartesianOSCEvalCfg):
    events: DataCollectionStateEventCfg = DataCollectionStateEventCfg()

@configclass
class Ur5eRobotiq2f85EvalStateRelCartesianOSCCfg(Ur5eRobotiq2f85StateRelCartesianOSCEvalCfg):
    events: StateEvalEventCfg = StateEvalEventCfg()
    scene: EvalStateSceneCfg = EvalStateSceneCfg(num_envs=32, env_spacing=1.5, replicate_physics=False)
    observations: EvalStateObservationsCfg = EvalStateObservationsCfg()
