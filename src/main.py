from PIL import Image
import torch
import torch.nn as nn
import numpy as np

from utils import transform_image, resize_image, scale_cam_image, show_cam_on_image, tensor_to_image, save_img
from coalition import ImagePlayerIterator
from dasp import DASP

# Constants
DEVICE = "cpu"
NUM_SHAPLEY_PLAYERS = 12  # Number of players for Shapley value computation
INPUT_IMAGE_PATH = "./example/input.png"
MODEL = "regnet"
OUTPUT_IMAGE_PATH = f"./example/{MODEL}-gups-output-image.png"
USE_GRADCAM = True
CAM_THRESHOLD = 0.5

# Load and preprocess input image
image = Image.open(INPUT_IMAGE_PATH).convert('RGB')
input_tensor = transform_image(image).unsqueeze(0)
input_numpy = tensor_to_image(input_tensor, is_batch=True)

model = None
# Load model
if MODEL == "mobilenet":
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights).to(DEVICE)
    model.eval()
elif MODEL == "efficientnet":
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights).to(DEVICE)
    model.eval()
elif MODEL == "regnet":
    from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights
    weights = RegNet_Y_400MF_Weights.DEFAULT
    model = regnet_y_400mf(weights=weights).to(DEVICE)
    model.eval()

indices = []
if USE_GRADCAM:
    from cam import get_grad_cam
    if MODEL == "regnet":
        target_layers = [model.trunk_output[-1]]
    else:
        target_layers = [model.features[-1]]
    grad_values = get_grad_cam(model, input_tensor, target_layers)
    # Get Grad-CAM mask above threshold
    mask = grad_values[0] > CAM_THRESHOLD
    rows, cols = np.where(mask)
    # Flatten the row, col indices
    indices = (rows * 7 + cols).tolist()

# Get feature map
prev_module = None
features_map = None
i = None
def hook_fn(module, input, output):
    global features_map
    features_map = output

modules = list(model.named_modules())
for i, (name, module) in reversed(list(enumerate(modules))):
    if isinstance(module, torch.nn.AdaptiveAvgPool2d) and module.output_size in {1, (1, 1)}:
        prev_name, prev_module = modules[i - 1]
        hook = prev_module.register_forward_hook(hook_fn)
        break

if i == 0:
    raise RuntimeError("No module found before AdaptiveAvgPool2d(1)")

model_outputs = model(input_tensor).softmax(dim=-1).cpu().detach().numpy()
predicted_class = np.argmax(model_outputs, axis=1)
if hook:
    hook.remove()

# Prepare for Shapley value calculation
image_tensor = features_map[0]
input_shape = image_tensor.shape

# Set up Shapley value calculation (not using Grad-CAM for now)
player_generator = ImagePlayerIterator(image_tensor, list_indices=indices, random=True, window_shape=(image_tensor.shape[0], 1, 1))
if MODEL == "efficientnet":
    linear_head = nn.Sequential(model.classifier[1])
    dasp = DASP(linear_head, player_generator=player_generator, input_shape=input_shape)
else:
    dasp = DASP(getattr(model, "classifier", None) or torch.nn.Sequential(getattr(model, "fc", None)), player_generator=player_generator, input_shape=input_shape)

# Run DASP for Shapley values
shapley_values = dasp.run(image_tensor, NUM_SHAPLEY_PLAYERS)

# Process Shapley values result for final visualization
cam_result = shapley_values[0, predicted_class[0]].detach().numpy()
cam_result = np.maximum(cam_result, 0)  # Remove negative values
cam_result = scale_cam_image(cam_result)
cam_result_resized = resize_image(cam_result, target_size=(224, 224))

# Overlay the CAM on the input image
final_cam_image = show_cam_on_image(input_numpy[0], cam_result_resized[0], use_rgb=True)
# Save the final output image
save_img(final_cam_image, OUTPUT_IMAGE_PATH)
