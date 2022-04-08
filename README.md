# Handwriting_Recognition_System
<p>
    <img src="https://img.shields.io/npm/l/color-calendar?style=flat-square" alt="license" />
</p>

<p>
A handwriting detection system using TensorFlow and a combination of Convolutional Recurrent Neural Network (CRNN) and Connectionist Temporal Classification (CTC) algorithms to train the RNN model. The RNN used in this project is Long Short Term Memory (LSTM) which is a type of RNN used specifically for image recognition. 
  
The effectiveness of the use of CTC for handwritten text recognition has been proven by Harald Scheidl in his thesis on "Handwritten Text Recognition in Historial documents." https://repositum.tuwien.at/retrieve/10807 
</p>

## QUICK DEMO

---
### Counting of Different Types of Vehicles:
<p align="center">
   <video src="https://user-images.githubusercontent.com/41968942/161442144-fa368bae-7d7f-4287-90ee-ad49dfefb818.mp4" | width=300>
</p>

- [Features](#features)
- [Installation](#Installation)
- [Usage](#usage)
  - [Main.py](#main)
  - [Model.py](#model)
  - [Preprocessor.py](#preprocess)
  - [Testing_code.py](#test)
  - [Training_model_code.py](#train)
  - [Create_lmdb.py](#lmdb)
  - [Dataloader_iam.py](#dataset)
- [Bug Reporting](#bug)
- [Feature Request](#feature-request)
- [Release Notes](#release-notes)
- [License](#license)

<a id="features"></a>

## 🚀 Features

- Zero dependencies
- Recognition of handwritten text on words (v.0.0.1)
- Recognition of handwritten text on lines (v.0.0.2) (incorporated)
- Use of CTC algorithms to improve the quality of recognition without using <a href="https://github.com/githubharald/DeslantImg">De-slanting</a> algorithms
- Implementation of different CTC decoders like 1) Best Fit 2) Beam Search 3) Word Search (upcoming)   
- [Request more features](#feature-request)...

<a id="Installation"></a>
  ## Installation
### Directly 
```bash
pip install -r requirements.txt
```
#### Or
### Use a Virtual Environment

```bash
    virtualenv <virtual_environment_name>
    
    cd <virtual_environment_name>\Scripts
    
    activate.bat
    
    pip install -r requirements.txt
```



<a id="usage"></a>
## 🔨 Usage
    
<a id="main"></a>
### 1) Main.py
*This file contains the code for Tkinter GUI which is used to test or predict the text. This takes images of handwritten text as input.*

<a id="model"></a>                           
### 2) Model.py
*This file contains the structure of the model that is used in this project.*
  
<a id="preprocess"></a>                           
### 3) Preprocessor.py
*This file handles the preprocessing that needs to be done. It downsizes the inputs to either a width of 128 or height of 32 (without affecting it's quality).* 
              
<a id="test"></a>                           
### 4) testing_code.py
*This file contains the "infer" method which is used to initialize the testing or prediction of text.*
  
<a id="train"></a>                           
### 5) Training_model_code.py
*This file contains all the methods which are needed to build the model. This can be invoked using a command line argument and arguments.* 
  
<a id="lmdb"></a>                           
### 6) Create_lmdb.py
*This file initializes the Lighting Memory Mapped Database (lmdb) which is used to store the images in a tree format.*
  
<a id="dataset"></a>                           
### 7) Dataloader_iam.py
*This file contains the import of <a href="https://fki.tic.heia-fr.ch/databases/iam-handwriting-database">IAM</a> dataset and it's preprocessing and split between test, train and validation data.*
  

<a id="bug"></a>

## 🐛 Bug Reporting

Feel free to [open an issue](https://github.com/deepanshug4/Vehicle_Classification_with_NumberPlate_Detection/issues) on GitHub if you find any bug.


<a id="feature-request"></a>

## ⭐ Feature Request

- Feel free to [Open an issue](https://github.com/deepanshug4/Handwriting_Recognition_GUI/issues) on GitHub to request any additional features you might need for your use case.
- Connect with me on [LinkedIn](https://www.linkedin.com/in/deepanshug4/). I'd love ❤️️ to hear where you are using this library.

<a id="release-notes"></a>

## 📋 Release Notes

Check [here](https://github.com/deepanshug4/Handwriting_Recognition_GUI/releases) for release notes.

<a id="license"></a>

## 📜 License

This software is open source, licensed under the [MIT License](https://github.com/deepanshug4/Handwriting_Recognition_GUI//blob/main/LICENSE).
