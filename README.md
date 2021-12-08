# Cassandre Project
*Made by : [Matthieu Bachelot](https://github.com/bachelow)*

# What is Project Cassandre ?

* Master thesis project
* Use deep learning methods on MRI volumes
* Detect tumors on MRI slice using semantic segmentation

# Methodology

* Test several architectures
* Test several data modeling / data generation methods
* Select the best

# Architectures 

* Auto Encoder / PSPnet / UNet / SegNet
* Kepp their architectures intact
* Test metric / loss / optimizers

# Data generation techniques
## Ressources constraints

![img/resources_constraints.PNG](img/resources_constraints.PNG)

# Data modeling methods
## Ground Truth representation

![img/ground_truth_representation.PNG](img/ground_truth_representation.PNG)

# Final results
## Focus on two networks -> reduce test time

![img/reduce_test_time.PNG](img/reduce_test_time.PNG)

![img/dice.PNG](img/dice.PNG)

![img/dice_custom.PNG](img/dice_custom.PNG)

* Statistical study on test data
* PSPNet was the best overall
* Training using Mini batch generation / One hot Vector representation / Data Augmentation

![img/final_loss_acuracy.PNG](img/final_loss_acuracy.PNG)

![img/dice_similatiry.PNG](img/dice_similatiry.PNG)

![img/box_plot.PNG](img/box_plot.PNG)

# Limits 

* Lot of result variation between patients
* Heterogeneous dataset
* time limits