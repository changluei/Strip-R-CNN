import torch

ckpt = torch.load("pretrained/strip_rcnn_s_dota_refine.pth", map_location="cpu")

new_state_dict = dict()

for key,value in ckpt["state_dict"].items():
    if "conv_spatial1" in key:
        key = key.replace("conv_spatial1", "strip_conv1")
    new_state_dict[key] = value

ckpt["state_dict"] = new_state_dict

torch.save(ckpt, "pretrained/strip_rcnn_s_dota.pth")