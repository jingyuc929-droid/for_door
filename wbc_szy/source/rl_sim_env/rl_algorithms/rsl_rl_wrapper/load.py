import onnx
import onnxruntime as ort


def verify_onnx_model(model_path, name):
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"{name} ONNX validation passed!")
    except onnx.checker.ValidationError as e:
        print(f"{name} ONNX validation failed: {e}")


def load_onnx_model(model_path):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
    return session


def onnx_run_inference(session, actor_obs, vae_obs):
    inputs = {
        "actor_obs": actor_obs,
        "vae_obs": vae_obs,
    }
    outputs = session.run(None, inputs)

    return outputs


def onnx_run_inference_amp_vae_vit(session, actor_obs, prop_his_obs, point_his_obs, h_prev):
    inputs = {
        "actor_obs": actor_obs,
        "prop_t": prop_his_obs,
        "point_t": point_his_obs,
        "h_prev": h_prev,
    }
    outputs = session.run(None, inputs)
    return outputs


def onnx_run_inference_locomotion(session, actor_obs, vae_obs):
    inputs = {
        "actor_obs": actor_obs,
        "vae_obs": vae_obs,
    }
    outputs = session.run(None, inputs)
    outputs_dict = {
        "actions": outputs[0],
    }
    return outputs_dict


def onnx_run_inference_MARGlocomotion(session, estimator_net_obs, actor_obs, gt_heightmap_obs):
    inputs = {
        "estimator_net_obs": estimator_net_obs,
        "actor_obs": actor_obs,
        "gt_heightmap_obs": gt_heightmap_obs,
    }
    outputs = session.run(None, inputs)
    outputs_dict = {
        "actions": outputs[0],
    }
    return outputs_dict


def onnx_run_inference_PIElocomotion(session, actor_obs, PIE_estimator_net_proprioceptive_obs, PIE_estimator_net_depth_images_obs, hidden_states):
    inputs = {
        "actor_obs": actor_obs,
        "PIE_estimator_net_proprioceptive_obs": PIE_estimator_net_proprioceptive_obs,
        "PIE_estimator_net_depth_images_obs": PIE_estimator_net_depth_images_obs,
        "hidden_states": hidden_states,
    }
    outputs = session.run(None, inputs)
    outputs_dict = {
        "actions": outputs[0],
        "new_hidden_states": outputs[1],
    }
    return outputs_dict