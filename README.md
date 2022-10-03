The project was developed for the Statistical Methods for Machine learning exam from the Data Science for Economics MSc at Unimi. 

The dataset used in this project is from real patients provided by the Policlinico of Milan, and the project is part of a collaboration with the EveryWare lab at the University of Milan. 
Furthermore, access to the lab's server and GPU allowed this project to conduct the numerous CNN experiments, which would be challenging on regular laptops otherwise.
The authors of this report are grateful to EveryWare's kind support with their time, their resources, and their domain knowledge.

The work focusses on the problem of multiclass classification of human joints (Knee, Elbow, Ankle, Other) from a dataset of 8693 ultrasounds.
<img src="http://....jpg" width="200" height="200" />

An exploration on classifiying the angles from which the ultrasound was taken has also been produced with the aim of 
comparing different approaches (transfer learning and K-means clustering on features extracted by the convolutional structure of the CNN) to solve the task.

The best performing architecture, inspired by our literature review and the VGG design philosophy, achieves an average 0.895 Accuracy score on a 5-fold group-aware Cross Validation. 

<img src="http://....jpg" width="200" height="200" />

The work is inspired by a literature review on the following papers:
- Y. LeCun. Generalization and network design strategies. 1989
- Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D.
Jackel. Backpropagation applied to handwritten zip code recognition. 1989
- Y. Le Cun, B. Boser, J. S. Denker, R. E. Howard, W. Habbard, L. D. Jackel, and D. Hen-
derson. Handwritten Digit Recognition with a Back-Propagation Network. 1990
- Yann LeCun, L ́eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning
applied to document recognition. 1998
- Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep
convolutional neural networks. 2012
- Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale
image recognition. 2014