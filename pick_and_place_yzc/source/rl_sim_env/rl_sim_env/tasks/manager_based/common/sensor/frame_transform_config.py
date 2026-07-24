from isaaclab.sensors import FrameTransformerCfg


def create_body_frame_transform_cfg(robot_base_link: str, robot_foot_names: list[str]) -> FrameTransformerCfg:
    """Create a frame transform configuration for a robot body."""
    frame_trans = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + robot_base_link,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name=foot_name,
                prim_path="{ENV_REGEX_NS}/Robot/" + foot_name,
            )
            for foot_name in robot_foot_names
        ],
    )

    return frame_trans
