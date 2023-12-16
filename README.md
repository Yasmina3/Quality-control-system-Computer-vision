There is a python script `opencv_object_tracking.py` that allows the real-time usage of the developed models

Note: you do not have to run the jupyter notebook for local testing, since the modal is already trained and results are the TFLite models.


## Instructions before use

1- Clone repo using: `git clone <repo>`
2- If you want to run the jupyter notebbok, download the dataset from Kaggle: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product.
3- Unzip and Copy the data folder into the repository. The folder should be named `casting_data` and include the `test` and `train` folders.
4- Install the imported libraries in a new python virtual environment to avoid dependencies errors.
5- Note that running the notebook as it is, will override the saved models in the `saved_models` folder.

## How to use for local testing
1- Open terminal and change directory the folder cloned
2- Run the following command: `pip install requirements.txt` 
3- Run the following command: `python opencv_object_tracking.py --video casting_test.mp4 --model cnn_model_vs.tflite` -make sure that the name of the files in `saved_models`is `cnn_model_vs.tflite`, else change the name in the command.
4- The video `casting_test.mp4` -a video that was created from the test images of the data set- will start running and another window, containing the necessary information retrieved from the trained model will appear .
5- Press `s` to stop the video at a certain shot. Select the frame directly from the video that will be the input of the image classifier. Press `enter` or `space` in order to start the model. Now, for all the pictures of the mechanical component, we will get `label` which has the values [deficient, normal], and `probability`, which gives the certainty probability of the result. Besides, in the other window, the product number, time, label, and probability will be appearing continuously until the system stops.
6- To stop the system, press first `s` again to pause, and then press `c` to stop it completely. 

The trained TFLite models can be used in practice to embedded devices based on Linux, such as Raspberry Pi and Coral devices with Edge TPU, among many others.






