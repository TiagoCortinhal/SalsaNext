# SalsaNext
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/salsanext-fast-semantic-segmentation-of-lidar/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=salsanext-fast-semantic-segmentation-of-lidar)

## Abstract 

In this paper, we introduce SalsaNext for the semantic segmentation of a full 3D LiDAR point cloud in real-time. SalsaNext is the next version of SalsaNet [1] which has an encoder-decoder architecture where the en- coder unit has a set of ResNet blocks and the decoder part combines upsampled features from the residual blocks. In contrast to SalsaNet, we have an additional layer in the encoder and decoder, introduce the context module, switch from stride convolution to average pooling and also apply central dropout treatment. To directly optimize the Jaccard index, we further combine the weighted cross entropy loss with Lova ́sz-Softmax loss [2]. We provide a thorough quan- titative evaluation on the Semantic-KITTI dataset [3], which demonstrates that the proposed SalsaNext outperforms other state-of-the-art semantic segmentation networks in terms of accuracy and computation time.

## Introduction 

Scene understanding is an essential prerequisite for au- tonomous vehicles. Semantic segmentation helps gaining a rich understanding of the scene by predicting a meaningful class label for each individual sensory data point. Achieving such a fine-grained semantic prediction in real-time acceler- ates reaching the full autonomy to a great extent.
Advanced deep neural networks have recently had a quantum jump in generating accurate and reliable semantic segmentation with real-time performance. Most of these approaches, however, rely on the camera images [4], [5], whereas relatively fewer contributions have discussed the semantic segmentation of 3D LiDAR data [6], [7]. The main reason is that unlike camera images, LiDAR point clouds are relatively sparse, unstructured, and have non-uniform sampling although LiDAR scanners have wider field of view and return more accurate distance measurements.
As comprehensively described in [8], there exist two mainstream deep learning approaches addressing the seman- tic segmentation of 3D LiDAR data only: point-wise and projection-based neural networks (see Fig. 1). The former approaches operate directly on the raw 3D points without requiring any pre-processing step, whereas the latter project the point cloud into various formats such as 2D image view or high-dimensional volumetric representation. As illustrated in Fig. 1, there is a clear split between these two approaches in terms of accuracy, runtime and memory consumption.

For instance, projection-based approaches (shown in green circles in Fig. 1) achieve the state-of-the-art accuracy while running significantly faster. Although point-wise networks (red squares) have slightly less number of parameters, they cannot efficiently scale up to large point sets due to the limited processing capacity, thus, they take a longer runtime.
In this work, we introduce a new neural network to perform semantic segmentation of a full 3D LiDAR point cloud in real-time. Our proposed network is build upon the SalsaNet model [1], hence, named SalsaNext. The SalsaNet model has an encoder-decoder skeleton where the encoder unit consists of a series of ResNet blocks and the decoder part upsamples and fuses features extracted in the residual blocks. The here proposed SalsaNext incorporates the fol- lowing improvements over the SalsaNet version:
• To process the full 360◦ LiDAR scan, the network depth was increased by inserting additional layers in the encoder and decoder units.
• To capture the global context information, a new context module was introduced before the encoder unit.
• To boost the roles of very basic features (e.g. edges and curves) in the segmentation process, the dropout treatment was altered by omitting the first and last network layers in the dropout process.
• To have a lighter model, average pooling was employed instead of having stride convolutions in the encoder.
• To enhance the segmentation accuracy by optimizing
the mean intersection-over-union score, i.e. the Jaccard index, the weighted cross entropy loss in SalsaNet was combined with the Lova ́sz-Softmax loss [2].

## Example Video

[![Inference of Sequence 13](https://i.ytimg.com/vi/WEAaq7GWSz0/hqdefault.jpg?sqp=-oaymwEjCNACELwBSFryq4qpAxUIARUAAAAAGAElAADIQj0AgKJDeAE=&rs=AOn4CLBSdG0KOvr4nEYG8fT_CCuNhyUmgg)](https://www.youtube.com/watch?v=WEAaq7GWSz0 "SalsaNext")


## How to use the code

First create the anaconda env with:
```conda env create -f salsanext.yml``` then activate the environment with ```conda activate salsanext```.

To train/eval you can use the following scripts:


 * [Training script](train.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]``` : Path to the dataset
     * ```-a [String]```: Path to the Architecture configuration file 
     * ```-m [String]```: Which model to use (rangenet,salsanet,salsanext)
     * ```-l [String]```: Path to the main log folder
     * ```-n [String]```: additional name for the experiment
     * ```-c [String]```: GPUs to use
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
     * ```-n [String]```: Which model to use (rangenet,salsanet,salsanext)
   * If you want to infer&evaluate a model that you saved to ````/salsanext/logs/[the desired run]```` and you
   want to infer$eval only the validation and save the label prediction to ```/pred```:
     * ```./eval.sh -d /dataset -p /pred -m /salsanext/logs/[the desired run] -s validation -n salsanext```
     
     
     
 The model is defined [here](train/tasks/semantic/modules/segmentator.py), the training logic is
 [here](train/tasks/semantic/modules/trainer.py).
 
 
 ### References 
[1] E. E. Aksoy, S. Baci, and S. Cavdar, “Salsanet: Fast road and vehicle segmentation in lidar point clouds for autonomous driving,” CoRR, 2019. [Online]. Available: http://arxiv.org/abs/1909.08291

[2] M. Berman, A. Rannen Triki, and M. B. Blaschko, “The lova ́sz- softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks,” in CVPR, 2018, pp. 4413–4421.

[3] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stach- niss, and J. Gall, “SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences,” in ICCV, 2019.

[4] A. Kendall, V. Badrinarayanan, and R. Cipolla, “Bayesian segnet: Model uncertainty in deep convolutional encoder-decoder architectures for scene understanding,” arXiv preprint arXiv:1511.02680, 2015.

[5] R. P. K. Poudel, S. Liwicki, and R. Cipolla, “Fast-scnn: Fast semantic segmentation network,” CoRR, vol. abs/1902.04502, 2019. [Online]. Available: http://arxiv.org/abs/1902.04502

[6] B. Wu, A. Wan, X. Yue, and K. Keutzer, “Squeezeseg: Convolutional neural nets with recurrent crf for real-time road-object segmentation from 3d lidar point cloud,” ICRA, 2018.

[7] A. Milioto, I. Vizzo, J. Behley, and C. Stachniss, “RangeNet++: Fast and Accurate LiDAR Semantic Segmentation,” in IROS, 2019.

[8] Y. Guo, H. Wang, Q. Hu, H. Liu, L. Liu, and M. Bennamoun, “Deep learning for 3d point clouds: A survey,” CoRR, 2019. [Online]. Available: https://arxiv.org/abs/1912.12033
 
 
