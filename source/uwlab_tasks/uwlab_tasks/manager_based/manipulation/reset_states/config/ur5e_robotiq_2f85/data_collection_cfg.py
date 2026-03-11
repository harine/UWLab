from .rl_state_cfg import Ur5eRobotiq2f85RlStateCfg, RlStateSceneCfg, ObservationsCfg, make_insertive_object, make_receptive_object
from isaaclab.utils.configclass import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_tasks.manager_based.manipulation.reset_states.config.ur5e_robotiq_2f85.actions import (
    Ur5eRobotiq2f85RelativeOSCAction,
)
from uwlab_assets.robots.ur5e_robotiq_gripper import (
    EXPLICIT_UR5E_ROBOTIQ_2F85,
    IMPLICIT_UR5E_ROBOTIQ_2F85,
    Ur5eRobotiq2f85RelativeJointPositionAction,
)

from ... import mdp as task_mdp

@configclass
class DataCollectionRGBSceneCfg(RlStateSceneCfg):
    """Configuration for data collection scene."""
    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
        update_period=0,
        height=224,
        width=224,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.21290445, 0.4486344),
            rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=13.20
        )
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
        update_period=0,
        height=224,
        width=224,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8373904, 0.58778435, 0.28051114),
            rot=(0.29133367, 0.22761494, 0.51065357, 0.77624034),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=21.9
        )
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=224,
        width=224,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.55
        )
    )

@configclass
class DataCollectionObservationsCfg(ObservationsCfg):
    @configclass
    class PositionsCfg(ObsGroup):
        """Positions observations for policy group."""
        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset_metadata_key": "gripper_offset",
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )
        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )
        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class DataCollectionObservationsCfg(ObsGroup):
        """Observations for policy group."""
        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset_metadata_key": "gripper_offset",
                "root_asset_offset_metadata_key": "offset",
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_offset_metadata_key": "gripper_offset",
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    positions: PositionsCfg = PositionsCfg()
    data_collection: DataCollectionObservationsCfg = DataCollectionObservationsCfg()

@configclass
class DataCollectionRGBObservationsCfg(DataCollectionObservationsCfg):
    @configclass
    class CameraCfg(ObsGroup):
        front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224)
            },
        )

        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224)
            },
        )

        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224)
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    cameras: CameraCfg = CameraCfg()

@configclass
class EventCfg:
    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "base_paths": [
                f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/ObjectAnywhereEEAnywhere",
            ],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class DataCollectionCfg(Ur5eRobotiq2f85RlStateCfg):
    """Configuration for data collection."""
    events: EventCfg = EventCfg()
    observations: DataCollectionObservationsCfg = DataCollectionObservationsCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    # viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), lookat=(0.0, 0.0, 0.0), origin_type="world", env_index=0, asset_name="robot")
    # variants = variants

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass 
class DataCollectionRGBCfg(DataCollectionCfg):
    scene: DataCollectionRGBSceneCfg = DataCollectionRGBSceneCfg(num_envs=32, env_spacing=1.5)
    observations: DataCollectionRGBObservationsCfg = DataCollectionRGBObservationsCfg()
    # viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), lookat=(0.0, 0.0, 0.0), origin_type="world", env_index=0, asset_name="robot")
    # variants = variants

    def __post_init__(self):
        super().__post_init__()
