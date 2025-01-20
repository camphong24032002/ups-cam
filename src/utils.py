import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def resize_image(cam, target_size):
    result = []
    for img in cam:
        img = cv2.resize(img, target_size)
        result.append(img)
    np_result = np.float32(result)
    return np_result


def tensor_to_image(tensor, is_batch):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(tensor).numpy()
    if is_batch:
        img = np.transpose(img, (0, 2, 3, 1))
    else:
        img = np.transpose(img, (1, 2, 0))
    return img


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    max_cam = np.max(cam)
    cam = cam / max_cam
    heatmap = None
    max_cam = None
    return np.uint8(255 * cam)


def get_figname(cam, idx):
    return "./imgs/"+cam+str(idx)+".png"


def save_img(cam_img, fig_name):
    plt.clf()
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(cam_img)
    plt.savefig(fig_name)


def scale_cam_image(cam):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        result.append(img)
    np_result = np.float32(result)
    return np_result

def release_list(A):
    for i in range(len(A)):
        A[i] = None