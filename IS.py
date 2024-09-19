## inception score
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch import nn
from torchvision.models import inception_v3
import cv2
import multiprocessing
import numpy as np
import glob
import os
from scipy import linalg
from utility.datasetutil import *

from ignite.metrics import InceptionScore
import ignite.distributed as idist

from pytorch_image_generation_metrics import get_inception_score, get_fid



def preprocess_image(im):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        im: np.array, shape: (H, W, 3), dtype: float32 between 0-1 or np.uint8
    Return:
        im: torch.tensor, shape: (3, 299, 299), dtype: torch.float32 between 0-1
    """
    assert im.shape[2] == 3
    assert len(im.shape) == 3
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255
    im = cv2.resize(im, (299, 299))
    im = np.rollaxis(im, axis=2)
    im = torch.from_numpy(im)
    assert im.max() <= 1.0
    assert im.min() >= 0.0
    assert im.dtype == torch.float32
    assert im.shape == (3, 299, 299)

    return im


def preprocess_images(images, use_multiprocessing):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        images: np.array, shape: (N, H, W, 3), dtype: float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
    Return:
        final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
    """
    if use_multiprocessing:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            jobs = []
            for im in images:
                job = pool.apply_async(preprocess_image, (im,))
                jobs.append(job)
            final_images = torch.zeros(images.shape[0], 3, 299, 299)
            for idx, job in enumerate(jobs):
                im = job.get()
                final_images[idx] = im  # job.get()
    else:
        final_images = torch.stack([preprocess_image(im) for im in images], dim=0)
    assert final_images.shape == (images.shape[0], 3, 299, 299)
    assert final_images.max() <= 1.0
    assert final_images.min() >= 0.0
    assert final_images.dtype == torch.float32
    return final_images


def calculate_IS(images1, images2, use_multiprocessing, batch_size):
    """ Calculate IS between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch size: batch size used for inception network
    Returns:
        IS (scalar)
    """
    # images1 = preprocess_images(images1, use_multiprocessing)
    images2 = preprocess_images(images2, use_multiprocessing)

    IS, IS_std =  get_inception_score(images2)

    return IS


def load_images(image_paths,size):
    """ Loads all .png or .jpg images from a given path
    Warnings: Expects all images to be of same dtype and shape.
    Args:
        path: relative path to directory
    Returns:
        final_images: np.array of image dtype and shape.
    """

    first_image = cv2.imread(image_paths[0])
    W, H = first_image.shape[:2]
    image_paths.sort()
    image_paths = image_paths
    final_images = np.zeros((len(image_paths), size[1], size[0], 3), dtype=first_image.dtype)
    for idx, impath in enumerate(image_paths):
        im = cv2.imread(impath)
        im = cv2.resize(im,size)
        im = im[:, :, ::-1]  # Convert from BGR to RGB
        assert im.dtype == final_images.dtype
        final_images[idx] = im
    return final_images


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("--p1", "--path1", dest="path1",
                      help="Path to directory containing the real images")
    parser.add_option("--p2", "--path2", dest="path2",
                      help="Path to directory containing the generated images")
    parser.add_option("--multiprocessing", dest="use_multiprocessing",
                      help="Toggle use of multiprocessing for image pre-processing. Defaults to use all cores",
                      default=False,
                      action="store_true")
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Set batch size to use for InceptionV3 network",
                      type=int)

    options, _ = parser.parse_args()
    options.batch_size=4



    assert options.batch_size is not None, "--batch_size is an required option"

    paths1 = glob('Datasets/images/SD_DB1_original_aug/', '*/0_*')  # center image만 가져옴
    paths2 = glob('Datasets/images/SD_DB2_original_aug/', '*/0_*')  # center image만 가져옴

    Apath=paths1+paths2

    paths3 = glob('Datasets/images/HKPU_A_cyc', '*/0_*')  # center image만 가져옴
    paths4 = glob('Datasets/images/HKPU_B_cyc', '*/0_*')  # center image만 가져옴

    Bpath=paths3+paths4


    images1 = load_images(Apath,(180,70)) ### origin
    images2 = load_images(Bpath,(180,70)) ## genrated
    IS_SCORE = calculate_IS(images1, images2, options.use_multiprocessing, options.batch_size)
    print(IS_SCORE)