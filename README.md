# Bird Species Classification
This project was the final work to conclude the course "Complex Data Mining" offered by UNICAMP in 2023 (class MDC 013).

## Goal
The main task here was to create a Deep Learning model to process images from 200 different bird species and train it to build a classifier. The data used here is available on kaggle in the following link: https://www.kaggle.com/datasets/kedarsai/bird-species-classification-220-categories

## Approach
To accomplish the proposed task, the strategy chosen was to use the technique of transfer learning. This technique is based on using huge neural networks with weights previously trained on another big set of images. The idea of this architecture is to leverage a model that is already has a high accuracy level and then fine tune it to a specific problem. In the presente case, the choice was to preserve the whole network's pretrained wieghts frozen and only adapt the output layer to make it adequate to the problem. The network chosen here was Efficient Net B1, especially for 2 reasons:

* Computational Efficiency;
* Good level of accuracy in the Efficient Net versions baseline with imagenet dataset, that we can follow below:
<img width="427" height="345" alt="image" src="https://github.com/user-attachments/assets/25a7534c-3c29-411c-929d-d913378f75a2" />
Source: https://www.researchgate.net/figure/The-performance-of-the-EfficientNet-models-versus-other-CNNs-on-ImageNet-from-Tan-Lee_fig2_355191831

## Results


## Explainability
More than build a precise classifier, there was a strong need to understand how the model were making Its predictions. To obtain explanations for the inferences, GradCAM method was used. The GradCAM is a tool that can show in the images which features were more important for the classification process using a heatmap. This tool was particularly important in our case because we had a very complex problem with 200 classes in the output layer, so we had to have a very good comprehension of why the model was performing well on some classes and bad on others.

