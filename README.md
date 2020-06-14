# SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving

## Abstract 

In this paper, we introduce SalsaNext for the uncertainty-aware semantic segmentation of a full 3D LiDAR point cloud in real-time. SalsaNext is the next version of SalsaNet [1] which has an encoder-decoder architecture where the encoder unit has a set of ResNet blocks and the decoder part combines upsampled features from the residual blocks. In contrast to SalsaNet, we introduce a new context module, replace the ResNet encoder blocks with a new residual dilated convolution stack with gradually increasing receptive fields and add the pixel-shuffle layer in the decoder. Additionally, we switch from stride convolution to average pooling and also apply central dropout treatment. To directly optimize the Jaccard index, we further combine the weighted cross entropy loss with Lova ́sz-Softmax loss [2]. We finally inject a Bayesian treatment to compute the epistemic and aleatoric uncertainties for each point in the cloud. We provide a thorough quantitative evalua- tion on the Semantic-KITTI dataset [3], which demonstrates that the proposed SalsaNext outperforms other state-of-the- art semantic segmentation networks.
## Examples 
![Example Gif](/images/SalsaNext.gif)

### Video 
[![Inference of Sequence 13](https://img.youtube.com/vi/MlSaIcD9ItU/0.jpg)](http://www.youtube.com/watch?v=MlSaIcD9ItU)



### Semantic Kitti Segmentation Scores

| Approach             | Size           | car      | bicycle  | motorcycle | truck    | other-vehicle | person   | bicyclist | motorcyclist | road     | parking  | sidewalk | other-ground | building | fence    | vegetation | trunk    | terrain  | pole     | traffic-sign | mIoU     |
|----------------------|----------------|----------|----------|------------|----------|---------------|----------|-----------|--------------|----------|----------|----------|--------------|----------|----------|------------|----------|----------|----------|--------------|----------|
| PointNet[15]         | 50K points     | 46.3     | 1.3      | 0.3        | 0.1      | 0.8           | 0.2      | 0.2       | 0.0          | 61.6     | 15.8     | 35.7     | 1.4          | 41.4     | 12.9     | 31.0       | 4.6      | 17.6     | 2.4      | 3.7          | 14.6     |
| PointNet++[16]       | 50K points     | 53.7     | 1.9      | 0.2        | 0.9      | 0.2           | 0.9      | 1.0       | 0.0          | 72.0     | 18.7     | 41.8     | 5.6          | 62.3     | 16.9     | 46.5       | 13.8     | 30.0     | 6.0      | 8.9          | 20.1     |
| SPGraph[17]          | 50K points     | 68:3     | 0.9      | 4.5        | 0.9      | 0.8           | 1.0      | 6.0       | 0.0          | 49.5     | 1.7      | 24.2     | 0.3          | 68.2     | 22.5     | 59.2       | 27.2     | 17.0     | 18.3     | 10.5         | 20.0     |
| SPLATNet[22]         | 50K points     | 66.6     | 0.0      | 0.0        | 0.0      | 0.0           | 0.0      | 0.0       | 0.0          | 70.4     | 0.8      | 41.5     | 0.0          | 68.7     | 27.8     | 72.3       | 35.9     | 35.8     | 13.8     | 0.0          | 22.8     |
| TagentConv[36]       | 50K points     | 86.8     | 1.3      | 12.7       | 11.6     | 10.2          | 17.1     | 20.2      | 0.5          | 82.9     | 15.2     | 61.7     | 9.0          | 82.8     | 44.2     | 75.5       | 42.5     | 55.5     | 30.2     | 22.2         | 35.9     |
| RandLa-Net[37]       | 50K points     | **94.0** | 19.8     | 21.4       | **42.7** | **38.7**     | 47.5     | 48.8      | 4.6          | 90.4     | 56.9     | 67.9     | 15.5         | 81.1     | 49.7     | 78.3       | 60.3     | 59.0     | 44.2     | 38.1         | 50.3     |
| LatticeNet[23]       | 50K points     | 92.9     | 16.6     | 22.2       | 26.6     | 21.4          | 35.6     | 43.0      | **46.0**     | 90.0     | 59.4     | 74.1     | 22.0         | 88.2     | 58.8     | 81.7       | 63.6     | 63.1     | 51.9     | 48.4         |    52.9      |
| SqueezeSeg[6]        | 64x2048 pixels | 68.8     | 16.0     | 4.1        | 3.3      | 3.6           | 12.9     | 13.1      | 0.9          | 85.4     | 26.9     | 54.3     | 4.5          | 57.4     | 29.0     | 60.0       | 24.3     | 53.7     | 17.5     | 24.5         | 29.5     |
| SqueezeSeg-CRF[6]    | 64x2048 pixels | 68.3     | 18.1     | 5.1        | 4.1      | 4.8           | 16.5     | 17.3      | 1.2          | 84.9     | 28.4     | 54.7     | 4.6          | 61.5     | 29.2     | 59.6       | 25.5     | 54.7     | 11.2     | 36.3         | 30.8     |
| SqueezeSegV2[10]     | 64x2048 pixels | 81.8     | 18.5     | 17.9       | 13.4     | 14.0          | 20.1     | 25.1      | 3.9          | 88.6     | 45.8     | 67.6     | 17.7         | 73.7     | 41.1     | 71.8       | 35.8     | 60.2     | 20.2     | 36.3         | 39.7     |
| SqueezeSegV2-CRF[10] | 64x2048 pixels | 82.7     | 21.0     | 22.6       | 14.5     | 15.9          | 20.2     | 24.3      | 2.9          | 88.5     | 42.4     | 65.5     | 18.7         | 73.8     | 41.0     | 68.5       | 36.9     | 58.9     | 12.9     | 41.0         | 39.6     |
| RangeNet21[7]        | 64x2048 pixels | 85.4     | 26.2     | 26.5       | 18.6     | 15.6          | 31.8     | 33.6      | 4.0          | 91.4     | 57.0     | 74.0     | 26.4         | 81.9     | 52.3     | 77.6       | 48.4     | 63.6     | 36.0     | 50.0         | 47.4     |
| RangeNet53[7]        | 64x2048 pixels | 86.4     | 24.5     | 32.7       | 25.5     | 22.6          | 36.2     | 33.6      | 4.7          | **91.8** | 64.8     | 74.6     | 27.9         | 84.1     | 55.0     | 78.3       | 50.1     | 64.0     | 38.9     | 52.2         | 49.9     |
| RangeNet53++[7]      | 64x2048 pixels | 91.4     | 25.7     | 34.4       | 25.7     | 23.0          | 38.3     | 38.8      | 4.8          | **91.8** | **65.0** | 75.2     | 27.8         | 87.4     | 58.6     | 80.5       | 55.1     | 64.6     | 47.9     | 55.9         | 52.2     |
| 3d-MiniNet[27]       | 64x2048 pixels | 90.5     | 42.3     | **42.1**   | 28.5     | 29.4          | 47.8     | 44.1      | 14.5         | 91.6     | 64.2     | 74.5     | 25.4         | 89.4     | 60.8     | **82.8**   | 60.8     | **66.7** | 48.0     | 56.6         | 55.8     |
| SqueezeSegV3[24]         | 64x2048 pixels | 92.5     | 38.7     | 36.5       | 29.6     | 33.0          | 45.6     | 46.2      | 20.1         | 91.7     | 63.4     | 74.8     | 26.4         | 89.0     | 59.4     | 82.0       | 58.7     | 65.4     | 49.6     | 58.9         | 55.9    |
| SalsaNet[1]             | 64x2048 pixels | 87.5     | 26.2     | 24.6       | 24.0     | 17.5          | 33.2     | 31.1      | 8.4          | 89.7     | 51.7     | 70.7     | 19.7         | 82.8     | 48.0     | 73.0       | 40.0     | 61.7     | 31.3     | 41.9         | 45.4     |
| SalsaNext[Ours]            | 64x2048 pixels | 91.9     | **48.3** | 38.6       | 38.9     | 31.9          | **60.2** | **59.0**  | 19.4         | 91.7     | 63.7     | **75.8** | **29.1**     | 90.2     | **64.2** | 81.8       | 63.6     | 66.5     | **54.3** | **62.1**     | **59.5** |
| SalsaNext+Uncert[Ours]     | 64x2048 pixels | 91.6     | 40.7     | 26.0       | 28.2     | 24.4          | 53.7     | 54.1      | 12.1         | 91.1     | 63.1     | 74.9     | 25.1         | **90.4** | 62.5     | 82.3       | **64.0** | 66.5     | 53.5     | 56.1         | 55.8     |
 

## How to use the code

First create the anaconda env with:
```conda env create -f salsanext.yml``` then activate the environment with ```conda activate salsanext```.

To train/eval you can use the following scripts:


 * [Training script](train.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]``` : Path to the dataset
     * ```-a [String]```: Path to the Architecture configuration file 
     * ```-l [String]```: Path to the main log folder
     * ```-n [String]```: additional name for the experiment
     * ```-c [String]```: GPUs to use (default ```no gpu```)
     * ```-u [String]```: If you want to train an Uncertainty version of SalsaNext (default ```false```)
   * For example if you have the dataset at ``/dataset`` the architecture config file in ``/salsanext.yml``
   and you want to save your logs to ```/logs``` to train "salsanext" with 2 GPUs with id 3 and 4:
     * ```./train.sh -d /dataset -a /salsanext.yml -m salsanext -l /logs -c 3,4```
<br>
<br>

 * [Eval script](eval.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]```: Path to the dataset
     * ```-p [String]```: Path to save label predictions
     * ``-m [String]``: Path to the location of saved model
     * ``-s [String]``: Eval on Validation or Train (standard eval on both separately)
     * ```-u [String]```: If you want to infer using an Uncertainty model (default ```false```)
     * ```-c [Int]```: Number of MC sampling to do (default ```30```)
   * If you want to infer&evaluate a model that you saved to ````/salsanext/logs/[the desired run]```` and you
   want to infer$eval only the validation and save the label prediction to ```/pred```:
     * ```./eval.sh -d /dataset -p /pred -m /salsanext/logs/[the desired run] -s validation -n salsanext```
     
     
     

 
 ### References 
[1] E. E. Aksoy, S. Baci, and S. Cavdar, “Salsanet: Fast road and vehicle segmentation in lidar point clouds for autonomous driving,” in IEEE Intelligent Vehicles Symposium (IV2020), 2020.

[2] M. Berman, A. Rannen Triki, and M. B. Blaschko, “The lova ́sz- softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks,” in CVPR, 2018.

[3] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stach- niss, and J. Gall, “SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences,” in ICCV, 2019.

[4] A. Kendall, V. Badrinarayanan, and R. Cipolla, “Bayesian segnet: Model uncertainty in deep convolutional encoder-decoder architectures for scene understanding,” arXiv preprint arXiv:1511.02680, 2015.

[5] R. P. K. Poudel, S. Liwicki, and R. Cipolla, “Fast-scnn: Fast semantic segmentation network,” CoRR, vol. abs/1902.04502, 2019.

[6] B. Wu, A. Wan, X. Yue, and K. Keutzer, “Squeezeseg: Convolutional neural nets with recurrent crf for real-time road-object segmentation from 3d lidar point cloud,” ICRA, 2018.

[7] A. Milioto, I. Vizzo, J. Behley, and C. Stachniss, “RangeNet++: Fast and Accurate LiDAR Semantic Segmentation,” in IROS, 2019.

[8] Y. Guo, H. Wang, Q. Hu, H. Liu, L. Liu, and M. Bennamoun, “Deep learning for 3d point clouds: A survey,” CoRR, 2019.

[9] W. Shi, J. Caballero, F. Husza ́r, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, and Z. Wang, “Real-time single image and video super- resolution using an efficient sub-pixel convolutional neural network,” CoRR, vol. abs/1609.05158, 2016.

[10] B. Wu, X. Zhou, S. Zhao, X. Yue, and K. Keutzer, “Squeezesegv2: Improved model structure and unsupervised domain adaptation for road-object segmentation from a lidar point cloud,” in ICRA, 2019.

[11] Y. Wang, T. Shi, P. Yun, L. Tai, and M. Liu, “Pointseg: Real-time semantic segmentation based on 3d lidar point cloud,” CoRR, 2018.

[12] E. Shelhamer, J. Long, and T. Darrell, “Fully convolutional networks for semantic segmentation.” PAMI, 2016.

[13] C. Zhang, W. Luo, and R. Urtasun, “Efficient convolutions for real- time semantic segmentation of 3d point clouds,” in 3DV, 2018.

[14] O. Ronneberger, P.Fischer, and T. Brox, “U-net: Convolutional net-
works for biomedical image segmentation,” in Medical Image Com-
puting and Computer-Assisted Intervention, 2015, pp. 234–241.

[15] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in CVPR, 2017. [16] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” in NIPS,
2017.

[17] L. Landrieu and M. Simonovsky, “Large-scale point cloud semantic
segmentation with superpoint graphs,” in CVPR, 2018.

[18] C. R. Qi, W. Liu, C. Wu, H. Su, and L. J. Guibas, “Frustum pointnets
for 3d object detection from RGB-D data,” CoRR, 2017.

[19] Y. Zhou and O. Tuzel, “Voxelnet: End-to-end learning for point cloud
based 3d object detection,” in CVPR, 2018.

[20] L. P. Tchapmi, C. B. Choy, I. Armeni, J. Gwak, and S. Savarese,
“Segcloud: Semantic segmentation of 3d point clouds,” in IEEE Intl.
Conf. on 3D Vision (3DV), 2017, p. 537–547.

[21] F. J. Lawin, M. Danelljan, P. Tosteberg, G. Bhat, F. S. Khan, and
M. Felsberg, “Deep projective 3d semantic segmentation,” CoRR,
2017. [Online]. Available: http://arxiv.org/abs/1705.03428

[22] H. Su, V. Jampani, D. Sun, S. Maji, E. Kalogerakis, M. Yang, and J. Kautz, “Splatnet: Sparse lattice networks for point cloud
processing,” in CVPR, 2018.

[23] R.AlexandruRosu,P.Schu ̈tt,J.Quenzel,andS.Behnke,“LatticeNet:
Fast Point Cloud Segmentation Using Permutohedral Lattices,” arXiv
e-prints, p. arXiv:1912.05905, Dec. 2019.

[24] C. Xu, B. Wu, Z. Wang, W. Zhan, P. Vajda, K. Keutzer, and
M. Tomizuka, “Squeezesegv3: Spatially-adaptive convolution for effi-
cient point-cloud segmentation,” 2020.

[25] Y. Zeng, Y. Hu, S. Liu, J. Ye, Y. Han, X. Li, and N. Sun, “Rt3d:
Real-time 3-d vehicle detection in lidar point cloud for autonomous
driving,” IEEE RAL, vol. 3, no. 4, pp. 3434–3440, Oct 2018.

[26] M. Simon, S. Milz, K. Amende, and H. Gross, “Complex-yolo: Real-
time 3d object detection on point clouds,” CoRR, 2018.

[27] I.Alonso,L.Riazuelo,L.Montesano,andA.C.Murillo,“3d-mininet: Learning a 2d representation from point clouds for fast and efficient
3d lidar semantic segmentation,” 2020.

[28] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation:
Representing model uncertainty in deep learning,” in international
conference on machine learning, 2016, pp. 1050–1059.

[29] D.Feng,L.Rosenbaum,andK.Dietmayer,“Towardssafeautonomous driving: Capture uncertainty in the deep neural network for lidar 3d
vehicle detection,” in ITSC. IEEE, 2018, pp. 3266–3273.

[30] E. Ilg, O. Cicek, S. Galesso, A. Klein, O. Makansi, F. Hutter, and T. Brox, “Uncertainty estimates and multi-hypotheses networks for
optical flow,” in ECCV, 2018, pp. 652–667.

[31] B. Zhang and P. Wonka, “Point cloud instance segmentation using
probabilistic embeddings,” CoRR, 2019.

[32] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
image recognition,” in CVPR, 2016, pp. 770–778.

[33] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolu-
tional networks,” CoRR, vol. abs/1311.2901, 2013.

[34] X. Li, S. Chen, X. Hu, and J. Yang, “Understanding the disharmony between dropout and batch normalization by variance shift,” arXiv
preprint arXiv:1801.05134, 2018.

[35] Y. Gal, J. Hron, and A. Kendall, “Concrete dropout,” in Advances in
Neural Information Processing Systems, 2017, pp. 3581–3590.

[36] M. Tatarchenko, J. Park, V. Koltun, and Q. Zhou, “Tangent convolu-
tions for dense prediction in 3d,” in CVPR, 2018.

[37] Q. Hu, B. Yang, L. Xie, S. Rosa, Y. Guo, Z. Wang, N. Trigoni, and
A. Markham, “Randla-net: Efficient semantic segmentation of large-
scale point clouds,” 2019.

[38] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous
driving? the kitti vision benchmark suite,” in CVPR, 2012.
 
 
