import astra
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
from main_confMatrix import calcScore as calc_score

def algebraic_reconstruction(image, angles, show=False, binary=True):
    # angles = np.array([0, 90, 130, 160], dtype=np.float32)
    image_size = image.shape[0]
    image = (image - image.min()) / (image.max() - image.min())

    # create geometries and projector
    print("Creating geometries and projector")
    proj_geom = astra.create_proj_geom('parallel', 1.0, max(image.shape[0], image.shape[1]), angles)
    vol_geom = astra.create_vol_geom(image.shape[0],image.shape[1])
    proj_id = astra.create_projector('linear', proj_geom, vol_geom)

    # generate phantom image
    print("Generating phantom image")
    V_exact_id, V_exact = astra.data2d.shepp_logan(vol_geom)
    print(V_exact.shape, V_exact.min(), V_exact.max(), V_exact.dtype)
    V_exact = image
    print(V_exact.shape, V_exact.min(), V_exact.max(), V_exact.dtype)

    # create forward projection
    print("Creating forward projection")
    sinogram_id, sinogram = astra.create_sino(V_exact, proj_id)

    # reconstruct
    print("Reconstructing")
    recon_id = astra.data2d.create('-vol', vol_geom, 0)
    cfg = astra.astra_dict('SIRT')
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    cfg['option'] = { 'MinConstraint': 0, 'MaxConstraint': 1 }
    sirt_id = astra.algorithm.create(cfg)
    astra.algorithm.run(sirt_id, 200)
    V = astra.data2d.get(recon_id)

    # thresholding
    if binary:
        V[V<0.5] = 0
        V[V>=0.5] = 1

    # show
    if show:
        plt.gray()
        plt.imshow(V)
        plt.show()

    # garbage disposal
    #astra.data2d.delete([sinogram_id, recon_id, V_exact_id])
    #astra.data2d.delete([sinogram_id, recon_id])
    #astra.projector.delete(proj_id)
    #astra.algorithm.delete(sirt_id)
    return V

# 1st image
if True:
    pass
if False:
    pass
if False:
    for i in [90, 30]:
        print(f"1st image ({i})")
        path = "figures/1.png"
        angles = np.arange(0, i, 0.5)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = algebraic_reconstruction(image, angles, show=False, binary=True)
        # save image
        save_path = f"figures/sirt_figures/1_{i}.png"
        image = (image * 255).astype(np.uint8)
        cv2.imwrite(save_path, image)
if False:
    for i in [90, 30]:
        print(f"2st image ({i})")
        path = "figures/2.png"
        angles = np.arange(0, i, 0.5)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = algebraic_reconstruction(image, angles, show=False, binary=True)
        # save image
        save_path = f"figures/sirt_figures/2_{i}.png"
        image = (image * 255).astype(np.uint8)
        cv2.imwrite(save_path, image)
if False:
    for i in [90, 30, 20, 10]:
        print(f"3rd image ({i})")
        path = "figures/3.png"
        angles = np.arange(0, i, 0.5)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = algebraic_reconstruction(image, angles, show=False, binary=True)
        # save image
        save_path = f"figures/sirt_figures/3_{i}.png"
        image = (image * 255).astype(np.uint8)
        cv2.imwrite(save_path, image)
if False:
    for i in [90, 30, 20, 10]:
        print(f"4th image ({i})")
        path = "figures/4.png"
        angles = np.arange(0, i, 0.5)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = algebraic_reconstruction(image, angles, show=False, binary=True)
        # save image
        save_path = f"figures/sirt_figures/4_{i}.png"
        image = (image * 255).astype(np.uint8)
        cv2.imwrite(save_path, image)
if False:
    for i in [90]:
        print(f"5th image ({i})")
        path = "figures/5.png"
        angles = np.arange(0, i, 0.5)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = algebraic_reconstruction(image, angles, show=False, binary=False)
        # save image
        save_path = f"figures/sirt_figures/5_{i}.png"
        image = (image * 255).astype(np.uint8)
        cv2.imwrite(save_path, image)


for i in [90, 30, 20, 10]:
    for j in [3, 4]:
        predicted_segmentation = f"figures/sirt_figures/{j}_{i}.png"
        ground_truth_segmentation = f"figures/{j-2}.png"
        score = calc_score(predicted_segmentation, ground_truth_segmentation)
        print(i, j, score)