import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import argparse
import math



def poisson_blend(im_src, im_tgt, im_mask, center):
    # Extract dimensions
    mask_h, mask_w = im_mask.shape

    # Calculate blending region coordinates
    top_y = center[1] - mask_h // 2
    bottom_y = math.ceil(center[1] + mask_h / 2)
    left_x = center[0] - mask_w // 2
    right_x = math.ceil(center[0] + mask_w / 2)

    # Create the sparse matrix for the Laplacian operator
    laplace_matrix_1d = build_1d_laplacian(mask_w)
    laplace_matrix_2d = sp.block_diag([laplace_matrix_1d] * mask_h).tolil()
    laplace_matrix_2d.setdiag(-1, mask_w)
    laplace_matrix_2d.setdiag(-1, -mask_w)

    # Adjust matrix for boundary pixels in the mask
    for y in range(1, mask_h - 1):
        for x in range(1, mask_w - 1):
            if im_mask[y, x] == 0:  
                idx = x + y * mask_w

                laplace_matrix_2d[idx, idx] = 1
                laplace_matrix_2d[idx, idx + 1] = 0
                laplace_matrix_2d[idx, idx - 1] = 0
                laplace_matrix_2d[idx, idx + mask_w] = 0
                laplace_matrix_2d[idx, idx - mask_w] = 0

    laplace_operator = laplace_matrix_2d.tocsc()

    # Prepare the blended image result
    output_img = np.copy(im_tgt)

    # Flatten the mask for linear operations
    mask_flattened = im_mask.flatten()

    # Blend each color channel
    for c in range(im_src.shape[2]):
        source_flat = im_src[:, :, c].flatten()
        target_patch_flat = im_tgt[top_y:bottom_y, left_x:right_x, c].flatten()

        b_vector = laplace_operator.dot(source_flat)
        b_vector[mask_flattened == 0] = target_patch_flat[mask_flattened == 0]

        solved_values = spsolve(laplace_operator, b_vector)
        solved_values = np.clip(solved_values, 0, 255).astype(np.uint8)

        reshaped_result = solved_values.reshape(mask_h, mask_w)

        output_img[top_y:bottom_y, left_x:right_x, c] = np.where(
            im_mask == 255, reshaped_result, im_tgt[top_y:bottom_y, left_x:right_x, c]
        )

    return output_img


def build_1d_laplacian(width):
    """Construct the 1D Laplacian matrix for a given width."""
    laplacian_1d = sp.lil_matrix((width, width))
    laplacian_1d.setdiag(-1, -1)
    laplacian_1d.setdiag(4)
    laplacian_1d.setdiag(-1, 1)
    return laplacian_1d


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/llama.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


