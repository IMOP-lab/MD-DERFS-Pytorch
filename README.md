# MD-DERFS-Pytorch
**Official code implementation of Multidimensional Directionality-Enhanced Segmentation via large vision model**

### [Project page](https://github.com/IMOP-lab/MD-DERFS-Pytorch) | [Our laboratory home page](https://github.com/IMOP-lab) 

<div align=center>
  <img src="https://github.com/IMOP-lab/MD-DERFS-Pytorch/blob/main/figures/frame.png">
</div>
<p align=center>
  Fig. 1: The overall framework of MD-DERFS, where fundus OCT images are first processed by a pre-trained encoder of SAM to generate image embeddings. These embeddings are then separately passed through MFU and CDIN modules, followed by feature fusion via iAFF to yield the segmentation results. The loss is computed using LHMSE to update the parameters.
</p>

## Method

### MFU
<div align=center>
  <img src="https://github.com/IMOP-lab/MD-DERFS-Pytorch/blob/main/figures/MFU.png"width=80% height=80%>
</div>
<p align=center>
  Fig. 2: Diagram of the MFU network structure. The MFU employs feature slicing to segment the input into distinct groups. Each group is processed by the Edema Texture Mapping Unit to extract directional prior features. Subsequent fusion via the iAFF optimizes segmentation accuracy while maintaining the framework's complexity at a manageable level.
</p>

### ETMU
<div align=center>
  <img src="https://github.com/IMOP-lab/MD-DERFS-Pytorch/blob/main/figures/ETMU.png"width=40% height=40%>
</div>
<p align=center>
  Fig. 3: The Edema Texture Mapping Unit mentioned in the overview, where LN means Layer Normalization, the MFU's sliced features are fed into the Edema Texture Mapping Unit to achieve a comprehensive multi-angle analysis of the image in terms of spatial and textural nuances.
</p>

### CDIN
<div align=center>
  <img src="https://github.com/IMOP-lab/MD-DERFS-Pytorch/blob/main/figures/CDIN.png"width=80% height=80%>
</div>
<p align=center>
  Fig. 4: The network structure diagram of CDIN, where image embeddings obtained from the SAM encoder, along with e-MFU and e-Final inputs, are fed into CDIN. The network features multiple stages of deep feature extraction using AAD modules and MLFA modules. The architecture includes upsampling layers to refine feature maps, ReLU activation functions for non-linearity, and skip connections for improved information flow. The final output is achieved through the sequential processing and combination of features from AAD and MLFA modules.
</p>

### MLFA
<div align=center>
  <img src="https://github.com/IMOP-lab/MD-DERFS-Pytorch/blob/main/figures/MLFA.png">
</div>
<p align=center>
  Fig. 5: The network structure diagram of MLFA. The module consists of two stages, each incorporating a dilated convolution layer followed by a BN layer and a ReLU activation function. Between these stages, an upsampling layer is employed to increase the spatial resolution of the feature maps. This architecture aims to capture multi-scale features effectively by leveraging dilated convolutions and subsequent upsampling. 
</p>

### AAD
<div align=center>
  <img src="https://github.com/IMOP-lab/MD-DERFS-Pytorch/blob/main/figures/AAD.png"width=80% height=80%>
</div>
<p align=center>
  Fig. 6: The network structure diagram of AAD. The model consists of two main components: the Scene Encoder and the Content Encoder. The Scene Encoder primarily includes two dilated convolutions and an SPP module. The Content Encoder comprises dilated convolutions, BN layers, and ReLU activation functions. The outputs from these encoders are combined and processed through a sigmoid function to form relational features, which are then integrated to produce the final P-feature.
</p>

## Experiments and Results
**Here, we present the quantitative and qualitative results of different benchmark models**

## Quantitative Results
<div align=center>
  <img src="https://github.com/IMOP-lab/MD-DERFS-Pytorch/blob/main/tables/baseline_table.png"width=80% height=80%>
</div>
<p align=center>
  Table 1: Experimental results of the proposed method and 17 previous segmentation methods on the MacuScan-8k dataset. The best value of the experimental results is highlighted in red, and the second best value is highlighted in blue.
</p>

## Qualitative Results
<div align=center>
  <img src="https://github.com/IMOP-lab/MD-DERFS-Pytorch/blob/main/figures/baseline_visualize.png"width=80% height=80%>
</div>
<p align=center>
  Fig. 7: Visual comparison of the evaluation results of MD-DERFS and 17 other segmentation methods on the MacuScan-8k dataset. We selected 6 representative images for display. In these images, GroundTruth is shown in red, the mask of the model segmentation is shown in blue, and the coincident parts are white to indicate that the segmentation is correct. In instances F1 and F5, the edema regions are indistinguishably segmented from the adjacent retinal tissue, highlighting MD-DERFS's exceptional inferential strength for precise segmentation. Additionally, MD-DERFS consistently achieves accurate segmentation of the edema areas and their peripheries in instances F2, F3, F4, and F6.
</p>
