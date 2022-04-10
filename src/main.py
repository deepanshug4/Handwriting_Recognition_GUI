import tkinter as tk
from tkinter import ttk
from tkinter import font
from tkinter.filedialog import askopenfile

from deep_translator import GoogleTranslator # the open source translation library

import testing_code as test # file having the infer function used to test

model_type = 1 # for line model
decoder_type = 1 # for beam search
# loading the line model beforehand to decrease the run time
model_line_beam = test.Model(model_type, test.char_list_from_file(), decoder_type, must_restore=True) 

root = tk.Tk()
root.geometry('500x600') # The size of tkinter window
root.title('Handwritten Text Prediction and Translation')

primary_label = ttk.Label(root, text="Choose the File for Text Recognition", font= ("Helvetica", 20)).pack()

lang = tk.StringVar() # variable to store the translation language specified by the user

def get_language(text): # to get translations in different languages
    frame_trans = tk.Frame(root)
    frame_trans.pack()
    frame_trans_text = tk.Frame(frame_trans)
    frame_trans_text.pack()

    to_translate = text
    translated = GoogleTranslator(source='auto', target=lang.get()).translate(to_translate) # function to get the translated text
    
    ttk.Label(frame_trans_text, 
			  text = "Translated Text: ", 
			  font= ("Helvetica", 15)).pack(expand=True, fill='both', side='left', ipadx=10, ipady=10)

    ttk.Label(frame_trans_text, 
			  text = translated, 
			  font= ("Times New Roman", 15)).pack(expand=True, fill='both', side='left')

    try_again_btn = ttk.Button(frame_trans, text='Try again', command = lambda:frame_trans.destroy()) # to change the language
    try_again_btn.pack(ipadx=10, ipady=10, pady=5)



def open_file():
    file = askopenfile(filetypes =[('Image Files', '')]) # open the text file
    img = file.name
    predicted_text, probability = test.infer(model_line_beam, img, type_of_model=model_type) #infer function
    
    frame_root = ttk.Frame(root) # primary frame
    frame_root.pack() 
    frame_text = ttk.Frame(frame_root) # frame for the recognised text
    frame_text.pack()
    frame_prob = ttk.Frame(frame_root) # frame for the probability
    frame_prob.pack()
    
    ttk.Label(frame_text, 
			  text = "Predicted Text: ", 
			  font= ("Helvetica", 20)).pack(expand=True, fill='both', side='left', ipadx=10, ipady=10)

    ttk.Label(frame_text, 
			  text = predicted_text, 
			  font= ("Times New Roman", 20)).pack(expand=True, fill='both', side='left')

    ttk.Label(frame_prob, 
			  text = "Probability: ", 
			  font= ("Helvetica", 20)).pack(expand=True, fill='both', side='left', ipadx=10, ipady=10)

    ttk.Label(frame_prob, 
			  text = "{0:.3f}".format(probability*100), 
			  font= ("Times New Roman", 20)).pack(expand=True, fill='both', side='left')

    try_again_btn = ttk.Button(frame_root, text='Try again', command = lambda:frame_root.destroy()) # to try for a new image
    try_again_btn.pack(ipadx=10, ipady=20, pady=5)

    ttk.Separator(frame_root,orient='horizontal').pack(fill='x', pady=10) 

    frame_translate = ttk.Frame(frame_root) # frame for text translation objects
    frame_translate.pack()

    ttk.Label(frame_translate, 
			  text = "Type the language", 
			  font= ("Helvetica", 10)).pack(expand=True, fill='both', side='left', ipadx=10, ipady=10)

    ttk.Entry(frame_translate, 
			  text="Language:", 
			  textvariable=lang).pack(expand=True, fill='both', side='left', padx=5, pady=5) 
    
    convert_btn = ttk.Button(frame_root, text='convert', command= lambda:get_language(predicted_text))
    convert_btn.pack(ipadx=10, ipady=10, pady=5)


# to select the text image
open_btn = ttk.Button(root, text='Choose File', command=open_file)
open_btn.pack(ipadx=20, ipady=20, pady=10)

root.mainloop()
