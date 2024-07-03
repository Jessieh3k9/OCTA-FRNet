import torch

# 检查是否有 MPS 设备
mps_available = torch.backends.mps.is_available()

# 获取可用的 GPU 数量
count_card = torch.cuda.device_count()

# 获取可用设备总数（包括 MPS 和 CUDA）
device_options = []
if mps_available:
    device_options.append("mps")
if count_card > 0:
    for i in range(count_card):
        device_options.append(f"cuda:{i}")
device_options.append("cpu")


# # 设备选择逻辑
# if len(device_options) > 1:
#     while True:
#         print(f"Available devices:")
#         for i, device in enumerate(device_options):
#             print(f"{i}: {device}")
#         s = input(f"Please choose a device number (0-{len(device_options)-1}): ")
#         if s.isdigit():
#             chosen_device = int(s)
#             if chosen_device >= 0 and chosen_device < len(device_options):
#                 break
#         print("Invalid input!")
#         continue
# else:
#     chosen_device = 0

# 设置设备
# device_cuda = torch.device(device_options[chosen_device])

def set_device():

    device_cuda = torch.device(device_options[0])
    print(f"\n\nDevice {device_options[0]} will be used.")
    return device_cuda
if __name__ == '__main__':
    import torch
    from mmpretrain import get_model

    model = get_model('convnext-v2-atto_3rdparty-fcmae_in1k', pretrained=True)
    inputs = torch.rand(1, 3, 224, 224)
    out = model(inputs)
    print(type(out))
    # To extract features.
    feats = model.extract_feat(inputs)
    print(type(feats))