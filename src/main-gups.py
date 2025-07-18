from PIL import Image
import torch
import numpy as np

from utils import transform_image, resize_image, scale_cam_image, show_cam_on_image, tensor_to_image, save_img
from cam import get_grad_cam
from coalition import ImagePlayerIterator
from dasp import DASP
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# Constants
DEVICE = "cpu"
CAM_THRESHOLD = 0.7  # Threshold for Grad-CAM values
NUM_SHAPLEY_PLAYERS = 12  # Number of players for Shapley value computation
INPUT_IMAGE_PATH = "./example/input.png"
OUTPUT_IMAGE_PATH = "./example/gups-output-image.png"

# Load and preprocess input image
image = Image.open(INPUT_IMAGE_PATH).convert('RGB')
input_tensor = transform_image(image).unsqueeze(0)
input_numpy = tensor_to_image(input_tensor, is_batch=True)

# Load MobileNetV3 model
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
model.eval().to(DEVICE)

# Compute Grad-CAM
grad_values = get_grad_cam(model, input_tensor)
model_outputs = model(input_tensor).softmax(dim=-1).cpu().detach().numpy()
predicted_class = np.argmax(model_outputs, axis=1)
features_map = model.features(input_tensor).to('cpu')

# Get Grad-CAM mask above threshold
mask = grad_values[0] > CAM_THRESHOLD
rows, cols = np.where(mask)

# Flatten the row, col indices
flattened_indices = (rows * 7 + cols).tolist()

# Prepare for Shapley value computation
image_tensor = features_map[0]
input_shape = image_tensor.shape

# Set up Shapley value calculation
player_generator = ImagePlayerIterator(image_tensor, list_indices=flattened_indices, random=True, window_shape=(image_tensor.shape[0], 1, 1))
dasp = DASP(model.classifier, player_generator=player_generator, input_shape=input_shape)

# Run DASP for Shapley values
shapley_values = dasp.run(image_tensor, NUM_SHAPLEY_PLAYERS)

# Process Grad-CAM result for final visualization
cam_result = shapley_values[0, predicted_class[0]].numpy()
cam_result = np.maximum(cam_result, 0)  # Remove negative values
cam_result = scale_cam_image(cam_result)
cam_result[grad_values >= CAM_THRESHOLD] = grad_values[grad_values >= CAM_THRESHOLD]

# Resize CAM to match image size and overlay on input image
cam_result_resized = resize_image(cam_result, target_size=(224, 224))
final_cam_image = show_cam_on_image(input_numpy[0], cam_result_resized[0], use_rgb=True)

# Save the final output image
save_img(final_cam_image, OUTPUT_IMAGE_PATH)
