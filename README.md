# RGB-T-Salient-Object-Detection
This is a work which builds a multi-Modal Fusion Network for RGB-T Salient Object Detection.
In this paper, we propose a deep learning 
network for the multi-modal fusion of RGB-T modalities for salient object detection called 
cross-attention deep fusion network (CADFNet). In our model, we employ a two-stream multi-modal encoder serving as our backbone network for extracting special multi-modal features. The 
output is then passed on to a top-down parallel decoder that contains multiple receptive field 
blocks to learn and predict multi-modal features. After that, a cross-attention complementarity 
exploration module is proposed to enrich, enhance and refine multi-modal features by exploiting 
the complementarity between them and finally, a deep supervision progressive fusion module is 
proposed to fuse these multi-modal features for accurate SOD. In the end, we compare our model 
to other state of the art methods to see how our method performed. The training and testing were 
conducted on the 3 benchmark RGB-T datasets; VT5000, VT1000 & VT821. Extensive 
experiments done on these three benchmark datasets demonstrate that our model is an effective 
RGB-T SOD framework that outperforms the current state-of-the-art models, both quantitatively 
and qualitatively.

