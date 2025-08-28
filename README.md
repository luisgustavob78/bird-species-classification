# Bird Species Classification
This project was the final work to conclude the course "Complex Data Mining" offered by UNICAMP in 2023 (class MDC 013).

## Goal
The main task here was to create a Deep Learning model to process images from 200 different bird species and train it to build a classifier. The data used here is available on kaggle in the following link: https://www.kaggle.com/datasets/kedarsai/bird-species-classification-220-categories

## Approach
To accomplish the proposed task, the strategy chosen was to use the technique of transfer learning. This technique is based on using huge neural networks with weights previously trained on another big set of images. The idea of this architecture is to leverage a model that already has a high accuracy level and then fine tune it to a specific problem. In the presente case, the choice was to preserve the whole network's pretrained wieghts frozen and only adapt the output layer to make it adequate to the problem. The network chosen here was Efficient Net B1, especially for 2 reasons:

* Computational Efficiency;
* Good level of accuracy in the Efficient Net versions baseline with imagenet dataset, that we can see below:
<img width="427" height="345" alt="image" src="https://github.com/user-attachments/assets/25a7534c-3c29-411c-929d-d913378f75a2" />

Source: https://www.researchgate.net/figure/The-performance-of-the-EfficientNet-models-versus-other-CNNs-on-ImageNet-from-Tan-Lee_fig2_355191831  

Here is the code for the architecture:
```
# Load Efficient Net B! weights from imagenet
from tensorflow.keras.applications import EfficientNetB1
model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the pretrained layers
for layer in model.layers:
    layer.trainable = False

# Add pooling, dropout to prevent overfitting, and the final output layer with the size of our problem (200)
pooling = tf.keras.layers.GlobalAveragePooling2D()(model.output)
dropout_layer = tf.keras.layers.Dropout(0.4)(pooling)
output_layer = tf.keras.layers.Dense(200, activation="softmax")(dropout_layer)
frozen_model = tf.keras.Model(model.input, output_layer)
```

Notice that we use the functional API coding. This is a requisite for GradCAM.

## Computational Resources 
To be able to train this network and load the Efficient Net weights, we had to use High RAM GPUs from Google Colab.

## Results
The solution reached a final balanced accuracy of 0.78 on the test set. When we look to the training process, we see that at the final iterations, the model was going to a dangerous overfitting path, with an increasing distance between loss and val_loss.  
<img width="933" height="357" alt="image" src="https://github.com/user-attachments/assets/16639bfc-8434-4028-b14c-014ac66f09e7" />  

To avoid the overfitting, a callback to stop the training process before it happens was used:
```
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, min_delta=0.01),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                              patience=4)

]
```

## Explainability
More than build a precise classifier, there was a strong need to understand how the model was making Its predictions. To obtain explanations for the inferences, GradCAM method was used. The GradCAM is a tool that can show in the images which features were more important for the classification process using a heatmap. This tool was particularly important in our case because we had a very complex problem with 200 classes in the output layer, so we had to have a very good comprehension of why the model was performing well on some classes and bad on others.

Some species had a very poor results in terms of precision and recall, like Fish Crow. Taking this class as example, this is how the model is behaving:  
**Image 1**  
**Species:** Fish Crow  
**Model Prediction:** Common Raven  
**Prediction Score:** ~0.5   
<img width="368" height="500" alt="image" src="https://github.com/user-attachments/assets/b2f0bec6-0907-444d-b268-375dd95d9f6e" />  

**Image 2**  
**Species:** Fish Crow  
**Model Prediction:** American Crow  
**Prediction Score:** 0.28  
<img width="500" height="375" alt="image" src="https://github.com/user-attachments/assets/1d2db845-5a3f-4bc4-bab3-81f2d24c8a27" />  

These are two miss classified images from the same species, but with different meanings. In the Image 1, we can see that the model tried to extract some features from the body of the bird and then classified it with a medium level of confidence, which is dangerous because can lead us to more mistakes. On the other hand, in the image 2 the model tried to extract features from the tail and the prediction socre was much lower. It may be explained by the fact that, in this region of the body, many species might look similar.  

One species with good results was Common Raven. Here are a few examples:  
**Image 3**  
**Species:** Common Raven  
**Model Prediction:** Common Raven  
**Prediction Score:** 0.66  
<img width="500" height="333" alt="image" src="https://github.com/user-attachments/assets/e23d17ac-23e9-4ad0-b213-cb335180bf8f" />  

**Image 4**  
**Species:** Common Raven  
**Model Prediction:** Common Raven  
**Prediction Score:** 0.8  
<img width="500" height="334" alt="image" src="https://github.com/user-attachments/assets/799e69df-5fc1-4304-9971-01201eccce62" />  

In both cases the model classified it right with high levels of confidence in Its predictions. We can also see that the wings region was the most considered by the model as features to generate these inferecene, which can lead us to conclude that this part of the bird is a very good indication of which species It belongs to.  

Another interesting case was the following:  
**Image 5**  
**Species:** American Crow  
**Model Prediction:** American Crow  
**Prediction Score:** 0.45  
<img width="500" height="399" alt="image" src="https://github.com/user-attachments/assets/da188907-45a5-4360-ab54-572450af64ba" />  

Despite the fact that the model made a correct classification, this prediction is very weird because the model used features from other regions of the image instead of looking to the bird. But why did it choose American Crow as species? Maybe because this is the species with more smaples in the dataset? It would be something to be investigated.

## Reproducibility
To run the code of this solution, it is recommended to use GPUs in google colab with the already installed tensorflow and keras libraries. Besides that, it requires kaggle package to download the images as the following:  
```
#download the lib to split the folders in train and validation
!pip install split-folders

#kaggle package
!pip install kaggle

#create a kaggle directory
!mkdir ~/.kaggle

#save json with kaggle APIs
! cp kaggle.json ~/.kaggle/

#images download
! kaggle datasets download nishantbansal01/bird-species-classification

#unzip images folders
!unzip -q bird-species-classification.zip
```

You also need a token to be able to use kaggle API to download the images. In your kaggle profile:
settings -> API -> Generate New Token (It will trigger the download of kaggle.json file).

The downloaded file must be in your project folder.




