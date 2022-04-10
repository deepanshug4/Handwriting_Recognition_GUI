# Handwriting_Recognition_And_Translation_GUI
<p>
    <img src="https://img.shields.io/npm/l/color-calendar?style=flat-square" alt="license" />
</p>

<p>
This project is a part of my Research work titled "Text Recognition from Handwritten Text using CRNN Models, Connectionist Temporal Classification (CTC)". This handwriting detection system uses TensorFlow and a combination of Convolutional Recurrent Neural Network (CRNN) along with Connectionist Temporal Classification (CTC) algorithms to train the RNN model. The RNN used in this project is Long Short Term Memory (LSTM) which is a type of RNN used specifically for image recognition. During the research, the efficiency of CRNN models was found to be much better than the existing methods like Hidden Markov Models and Multi layer Perceptrons. In this project, I havce proposed a model which performs better than it's predecessors.

The effectiveness of the use of CTC for handwritten text recognition has been proven by Harald Scheidl in his thesis on "Handwritten Text Recognition in Historial documents." https://repositum.tuwien.at/retrieve/10807 

To make the project more accessible and useful an additional translation model is also used to get the functionality of converting the recognised text into a preferred language.
</p>

<p> This project can be used to train both word model and line model. Tho, line model supersedes the word model. Along with this the testing can be done with different types of decoders and the difference between their efficiencies can be observed.
The use of CTC makes the model more receptive to differnt writing styles like italics and cursive. 
</p>

## QUICK DEMO
---
### Counting of Different Types of Vehicles:
<p align="center">
   <video src="https://user-images.githubusercontent.com/41968942/162639093-34b066e6-fc49-47a5-9bd0-f5b1aeddf53a.mp4" | width=300>
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
- Use of CTC algorithms to improve the quality of recognition without using <a href="https://github.com/githubharald/DeslantImg">De-slanting</a> algorithms
- Recognition of handwritten text on words (v.0.0.1)
- Recognition of handwritten text on lines (v.0.0.2)
- Implementation of Translation Module on Recognised Text (v.0.0.3) (Latest)
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
*This file contains the code for Tkinter GUI which is used to test or predict the text along with translation of the recognised text. This takes images of handwritten text as input and predicts the text contained in it.*
```
    model_type = 1 # for line model
    decoder_type = 1 # for beam search
    # loading the line model beforehand to decrease the run time
    model_line_beam = test.Model(model_type, test.char_list_from_file(), decoder_type, must_restore=True) 
    
    def open_file():
    file = askopenfile(filetypes =[('Image Files', '')]) # open the text file
    img = file.name
    predicted_text, probability = test.infer(model_line_beam, img, type_of_model=model_type) #infer function
    ...
```
    
*For the translation part <a href="https://pypi.org/project/deep-translator/">deep_translator</a> library is used. This is an open-source library which has different modules for text translation. *
    
```
    from deep_translator import GoogleTranslator # the open source translation library
    def get_language(text): # to get translations in different languages
        frame_trans = tk.Frame(root)
        frame_trans.pack()
        frame_trans_text = tk.Frame(frame_trans)
        frame_trans_text.pack()

        to_translate = text
        translated = GoogleTranslator(source='auto', target=lang.get()).translate(to_translate) # function to get the translated text
        ...
```
    
<a id="model"></a>                           
### 2) Model.py
*This file contains the structure of the model that is used in this project. The model used in this project has 3 different components.*
<ul>
<li> CNN
    
```
    def setup_cnn(self) -> None:
        """Create CNN layers. ReLU activation function."""
        cnn_in4d = tf.expand_dims(input=self.input_imgs, axis=3)

        # list of parameters for the layers
        kernel_vals = [5, 5, 3, 3, 3] # 2 layers of 5X5 and 3 layers of 2X2.
        feature_vals = [1, 32, 64, 128, 128, 256] 
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)
        ...
```
    
<li> RNN
    
```
   def setup_rnn(self) -> None:
        """Create RNN layers."""
        rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=[2])

        # basic cells which is used to build RNN
        num_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True) for _ in
                 range(2)]  # 2 layers are used for RNN.
        ...
```
    
<li> CTC
    
```
    def setup_ctc(self):
        """CTC function for loss and decoder."""

        self.ctc_in_3d_tbc = tf.transpose(a=self.rnn_out_3d, perm=[1, 0, 2])
        # ground truth text as sparse tensor
        self.gt_texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                        tf.compat.v1.placeholder(tf.int32, [None]),
                                        tf.compat.v1.placeholder(tf.int64, [2]))
        ...
```
    
</ul
    
*Along with these there is a train function to train the model with specific changes like the decoder type and type of model: 1) The old word model. 2) The newly proposed line model (works on both words and line). This train function can be called from the functions given in <a href="https://github.com/deepanshug4/Handwriting_Recognition_GUI/blob/main/src/training_model_code.py">training_model_code.py</a>*

```
   def train_batch(self, batch: Batch) -> float:
        """Feed a batch into the NN to train it."""
        num_batch_elements = len(batch.imgs)
```
    
*The infer method in the models file is the method to test the model on any input. It uses the stored model checkpoints to build the complete model depending upon the selected decoder. This has options to load the checkpoints of the word and line model as per the user's needs. This function can be called from <a href="https://github.com/deepanshug4/Handwriting_Recognition_GUI/blob/main/src/testing_code.py">testing_code.py</a>*
    
```
    def infer_batch(self, batch: Batch, calc_probability: bool = False, probability_of_gt: bool = False):
        """Feed a batch into the NN to recognize the texts."""

        # decode, optionally save RNN output
        num_batch_elements = len(batch.imgs)
```
    
<a id="preprocess"></a>                           
### 3) Preprocessor.py
*This file handles the preprocessing that needs to be done. It downsizes the inputs to either a width of 128 or height of 32 for word images and width of 256 or height of 32 for line images (without affecting it's quality).* 
              
<a id="test"></a>                           
### 4) testing_code.py
*This file contains the "infer" method which is used to initialize the testing or prediction of text. This method has attributes for the type of model to be selected and the type of decoder to be used.*
  
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
