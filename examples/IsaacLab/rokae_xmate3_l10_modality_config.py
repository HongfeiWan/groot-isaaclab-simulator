from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


rokae_xmate3_l10_config = {
    # Video keys must match meta/modality.json.
    # Use the available head/front camera only. Do not declare a missing wrist
    # camera; keeping train/test modalities identical avoids a black/fake-view
    # distribution shift.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "ego_view",
        ],
    ),
    # observation.state dim=20:
    # [0:7) arm_joint_pos, [7:10) arm_eef_pos, [10:20) hand_joint_pos.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm_joint_pos",
            "arm_eef_pos",
            "hand_joint_pos",
        ],
        # Only apply sin/cos to angle-like joint states, not Cartesian xyz.
        sin_cos_embedding_keys=["arm_joint_pos", "hand_joint_pos"],
    ),
    # action dim=13:
    # [0:3) arm_eef_pos_target, [3:13) hand_joint_target.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "arm_eef_pos_target",
            "hand_joint_target",
        ],
        action_configs=[
            # The arm action is stored as an absolute XYZ target in rokae_base.
            # Use the current arm_eef_pos state as the reference for relative chunking.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="arm_eef_pos",
            ),
            # L10 hand targets are absolute canonical joint targets.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(rokae_xmate3_l10_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
