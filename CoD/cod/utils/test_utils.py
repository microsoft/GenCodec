
def instantiate_class(config):
    kwargs = config.get("init_args", {})
    class_module, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**kwargs)

def load_model(weight_dict, denoiser, prefix="ema_denoiser."):
    if "module" in weight_dict:
        state_dict = weight_dict["module"]
    elif "state_dict" in weight_dict:
        state_dict = weight_dict["state_dict"]
    else:
        state_dict = weight_dict

    for k, v in denoiser.state_dict().items():
        try:
            v.copy_(state_dict[prefix + k])
        except:
            print(f"Failed to copy {prefix + k} to denoiser weight")
    return denoiser