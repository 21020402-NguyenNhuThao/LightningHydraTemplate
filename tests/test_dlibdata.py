import hydra
import omegaconf
import torch
import pyrootutils
import matplotlib.pyplot as plt
import scipy.io
import cv2
import albumentations as A
# mat = scipy.io.loadmat("/home/nnthao/Documents/AI/LightHydra/lightning-hydra-template/bounding_boxes_afw.mat")
# print(mat)

root = pyrootutils.setup_root(__file__, pythonpath=True)
cfg = omegaconf.OmegaConf.load(root / "configs" / "data" /
                               "dlib.yaml")
datamodule = hydra.utils.instantiate(cfg)
datamodule.setup()
# train_dataloader = datamodule.train_dataloader()
datamodule.draw_batch()
# batch_image = next(iter(train_dataloader))
# def draw_batch(batch):
#     fig = plt.figure(figsize=(8,8))
#     key_points = batch_image['keypoints']
#     for i in range(len(batch["image"])):
#         ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
#         image = batch['image'][i]
#         # key_points = batch['keypoints'][i]
#         for j in range(68):
#             plt.scatter(key_points[j][0][i], key_points[j][1][i], s=10, marker='.', c='r')
#         # plt.scatter(key_points[:,:1,i],key_points[:,:2,i],s=10, marker='.', c='r')
#         plt.imshow(image)
#     plt.show()

# draw_batch(batch_image)

