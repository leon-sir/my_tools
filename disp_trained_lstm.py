import torch

actuator_network_path = "pretrained_actuator_nets/anydrive_v3_lstm.pt"
# actuator_network_path = "train_turbo_csv/fly_robot_LPV.pt"
model = torch.jit.load(actuator_network_path).to("cpu")


print("=== Model Type ===")
print(model)


print("\n=== Model Submodule Hierarchy ===")
for name, module in model.named_modules():
    print(f"Module name: {name}")
    print(f"Module type: {type(module)}")
    for param_name, param in module.named_parameters(recurse=False):
        print(f"  |-- Parameter: {param_name}, Shape: {param.shape}")
    print("-" * 40)


print("\n=== Model Parameters Overview ===")
for name, param in model.named_parameters():
    print(f"Parameter: {name}, Shape: {param.shape}")


print("\n=== Model Graph Structure (if available) ===")
print(model.graph)