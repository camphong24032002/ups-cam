from PIL import Image
import torch
import torch.nn as nn
import numpy as np

from utils import transform_image, resize_image, scale_cam_image, show_cam_on_image, tensor_to_image, save_img
from coalition import ImagePlayerIterator
from dasp import DASP
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# Constants
DEVICE = "cpu"
NUM_SHAPLEY_PLAYERS = 12
INPUT_IMAGE_PATH = "./example/input.png"
OUTPUT_IMAGE_PATH = "./example/eff-ups-output-image.png"

# Load and preprocess input image
image = Image.open(INPUT_IMAGE_PATH).convert('RGB')
input_tensor = transform_image(image).unsqueeze(0).to(DEVICE)
input_numpy = tensor_to_image(input_tensor, is_batch=True)

# Load EfficientNetV2-S model with pretrained weights
weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights).to(DEVICE)
model.eval()

# Forward pass to get prediction and intermediate features
with torch.no_grad():
    # Extract features (before the classifier)
    features_map = model.features(input_tensor)  # Shape: [1, C, H, W]
    output = model(input_tensor)
    probs = output.softmax(dim=-1).cpu().numpy()
    predicted_class = np.argmax(probs, axis=1)

# Prepare for Shapley value calculation
image_tensor = features_map[0]  # Remove batch dimension
input_shape = image_tensor.shape
print(input_shape)

# Set up DASP
player_generator = ImagePlayerIterator(
    image_tensor, list_indices=[], random=True,
    window_shape=(image_tensor.shape[0], 1, 1)
)

linear_head = nn.Sequential(model.classifier[1])
# Use model.classifier only for classification head
dasp = DASP(linear_head, player_generator=player_generator, input_shape=input_shape)

# Run DASP for Shapley values
shapley_values = dasp.run(image_tensor, NUM_SHAPLEY_PLAYERS)

# Process and visualize
cam_result = shapley_values[0, predicted_class[0]].detach().numpy()
cam_result = np.maximum(cam_result, 0)
cam_result = scale_cam_image(cam_result)
cam_result_resized = resize_image(cam_result, target_size=(224, 224))
final_cam_image = show_cam_on_image(input_numpy[0], cam_result_resized[0], use_rgb=True)

# Save result
save_img(final_cam_image, OUTPUT_IMAGE_PATH)
