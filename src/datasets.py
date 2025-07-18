# import os
# import torch
# from torch.utils.data import DataLoader, Subset
# from torchvision import datasets
# import torchvision.transforms as transforms

# class Dataset:
#     def __init__(self, path, name, batch_size=64, shuffle=False, get_sample=False) -> None:
#         self.path = os.path.join(path, name)
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225]),
#             ])
#         if name == "imagenet":
#             self.dataset = datasets.ImageNet(root=self.path,
#                                              split='val',
#                                              transform=self.transform)
#         elif name == "cifar100":
#             self.dataset = datasets.CIFAR100(root=self.path,
#                                              train=False,
#                                              download=True,
#                                              transform=self.transform)
#         elif name == "coco":
#             root_path = os.path.join(self.path, "val2017/val2017")
#             ann_path = os.path.join(self.path, "ann2017/annotations/instances_val2017.json")
#             self.dataset = datasets.CocoDetection(root=root_path,
#                                                   annFile=ann_path,
#                                                   transform=self.transform)
#         if get_sample:
#             generator = torch.manual_seed(42)
#             num_samples = 2000
#             indices = torch.randperm(len(self.dataset), generator=generator).tolist()[:num_samples]
#             subset_data = Subset(self.dataset, indices)
#             self.loader = DataLoader(subset_data, batch_size=batch_size, shuffle=True)
#         else:
#             self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)


from PIL import Image
import torchvision.transforms as transforms

# Đọc ảnh đầu vào
image_path = 'input.jpg'  # Thay bằng đường dẫn ảnh của bạn
image = Image.open(image_path).convert('RGB')

# Định nghĩa chuỗi transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Áp dụng transform
transformed_tensor = transform(image)
