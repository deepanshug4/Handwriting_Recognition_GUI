from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile

import testing_code as test # File having the infer function used to test.


model = test.Model(test.char_list_from_file(), 1, must_restore=True) #0 for Best Fit, 1 for Beam Seach Decofing algorithm.

root = Tk()
root.geometry('500x300') # The size of tkinter window.
root.title('Handwritten Word Prediction')

def open_file():
	'''This is used to get the file and create a new dynamic frame which will present the results.
	The frame containing the results is a dynamic frame which gets destroyed.
	The try again button destroys the frame and it is built anew.
	'''
	frame = Frame(root)
	frame.pack()
	file = askopenfile(filetypes =[('Image Files', '')])
	img = file.name
	a, b = test.infer(model, img) #infer function 
	

	frame1 = Frame(root)
	frame1.pack()

	pred_word = Label(frame1, text = 'Prediction \n ' + a + '\n\n', font = "100") # The word
	pred_word.pack()
	prob_word = Label(frame1, text = '  Probability \n  ' + str(b), font = "100")
	prob_word.pack()
	
	try_again_btn = Button(frame1, text ='Try Again', command = lambda:frame1.destroy(), width = 50)
	try_again_btn.pack(pady = 10)

open_btn = Button(root, text ='Open', command = open_file, width = 50)
open_btn.pack(side = TOP, pady = 10)


mainloop()
