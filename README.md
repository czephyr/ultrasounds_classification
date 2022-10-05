### Classification of Ultrasound Images of Joints using Convolutional Neural Networks
The project was developed for the Statistical Methods for Machine learning exam from the Data Science for Economics MSc at Unimi. 

The dataset used in this project is from real patients provided by the Policlinico of Milan, and the project is part of a collaboration with the EveryWare lab at the University of Milan. 
Furthermore, access to the lab's server and GPU allowed this project to conduct the numerous CNN experiments, which would be challenging on regular laptops otherwise.
The authors of this report are grateful to EveryWare's kind support with their time, their resources, and their domain knowledge.

The work focusses on the problem of multiclass classification of human joints (Knee, Elbow, Ankle, Other) from a dataset of 8693 ultrasounds.
<p align="center">
<img src="https://github.com/czephyr/msa_CNNproject/blob/main/Knee%20Ultrasound%20demonstration.png" width="500" height="300" />
</p>
  
An exploration on identify which of 3 standard cross-sections used by radiologists was used to capture the ultrasound images of knees has also been produced with the aim of comparing different approaches (transfer learning and K-means clustering) to solve the task.

The best performing architecture, inspired by our literature review and the VGG design philosophy, achieves an average 0.895 Accuracy score on a 5-fold group-aware Cross Validation. 

<p align="center">
<img src="https://github.com/czephyr/msa_CNNproject/blob/main/msaarch.drawio.png" width="650" height="350" />
</p>

It was found that training a CNN from scratch achieved slightly better overall accuracy of 0.919 accuracy with respect to 0.895 from transfer learning on identifying the knee cross-sections. K-Means clustering on features extracted by the convolutional structure of the CNN was only able to distinguish SQR (A) scans which are collected from the top of the patella and Femoral (C) which are collected with the probe rotated by 90 degrees; Medial (B white) and Lateral (B black), collected on the right and left of the patella, with the probe positioned at the same orientation produce very similar images that the clustering wasn't able to distinguish. 

<p align="center">
<img src="https://github.com/czephyr/msa_CNNproject/blob/main/kneeSides.png" width="350" height="450" />
</p>

The work is inspired by a literature review on the following papers:

>- Y. LeCun. Generalization and network design strategies. 1989
> - Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. 1989
> - Y. Le Cun, B. Boser, J. S. Denker, R. E. Howard, W. Habbard, L. D. Jackel, and D. Henderson. Handwritten Digit Recognition with a Back-Propagation Network. 1990
> - Yann LeCun, L ́eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. 1998
> - Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. 2012
> - Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. 2014
