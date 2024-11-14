'''
CPSC 383, F24
Python code to collect written digit data for Assignment 2
for testing on MNIST model

Author: Janet Leahy

'''

from tkinter import *
from PIL import Image, ImageTk, ImageDraw
from io import BytesIO
import numpy as np

instructions = "Draw the requested digit in the box below. You can click and drag the mouse slowly to draw on the canvas.\n Once you have drawn all symbols, the window will close automatically and export your drawings as both PNG files and as a (10,28,28) numpy array."

SCALE = 8
WIDTH = 28 * SCALE
HEIGHT = 28 * SCALE
PEN_SIZE = 8

NUM_COPIES = 1  # 1 copy of each digit
NUM_SYMBOLS = 10    # 10 digits in total
copy_index = 1
symbol_index = 1

im = None

capture = None
drawcapture = None
counter = None
symbollabel = None
symbolarea = None

x_vals = []
y_vals = []

def counter_text():
    global copy_index, symbol_index
    return f"Draw the digit {symbol_index - 1}"    


#State of mouse
mouse = "up"
def click(event):
    global mouse
    mouse = "down"
    event.widget.create_oval(event.x,event.y,event.x,event.y, width=PEN_SIZE)
    drawcapture.ellipse([event.x-(0.5*PEN_SIZE), event.y-(0.5*PEN_SIZE),event.x+(0.5*PEN_SIZE), event.y+(0.5*PEN_SIZE)], fill=(0,0,0)) 
    
def release(event):
    global mouse
    mouse = "up"
    
def motion(event):
    if mouse == "down":
        event.widget.create_oval(event.x,event.y,event.x,event.y, width=PEN_SIZE)
        drawcapture.ellipse([event.x-(0.5*PEN_SIZE), event.y-(0.5*PEN_SIZE),event.x+(0.5*PEN_SIZE), event.y+(0.5*PEN_SIZE)], fill=(0,0,0)) 

def clear(widget):
    global drawcapture
    # erase writing on both canvas and image to be captured
    widget.delete("all")
    drawcapture.rectangle([0,0,WIDTH,HEIGHT],(255,255,255))

def save(widget):
    # save as png, then reload, resize, and extract pixels and append to numpy array
    global capture, x_vals, y_vals, copy_index, symbol_index, symbollabel
    filename = f"digit{symbol_index-1}.png"
    capture.save(filename)
    
    with Image.open(filename) as im:
        resized = im.resize((28,28))
    
    pix = np.array(resized.convert("L"))    # convert to grayscale
    #print(pix)
    x_vals.append(pix)
    
    # if finished, save and close window
    if (symbol_index == NUM_SYMBOLS):
        total_array = np.asarray(x_vals)
        print(total_array.shape)
        np.save("digits.npy", total_array)
        root.destroy()
        print("Congrats! All ten digits have been saved as PNG images and the pixel data has been exported as digits.npy")
        exit()
    
    # increment to next digit
    symbol_index += 1
    
    # reset canvas and update label
    clear(widget)
    
    global counter
    counter.config(text=counter_text())


class Window:
    def __init__(self, master):
        global counter, symbollabel, instructions, symbolarea
        
        instructions = Label(master, text = instructions)
        instructions.pack(padx = 20, pady = 20, side = TOP)
        counter = Label(master, foreground = "blue", text = counter_text())
        counter.pack(padx = 20, pady = 20, side = TOP)
        
        drawframe = Frame(master, background = "green")
        
        # create canvas to display drawing
        drawarea = Canvas(drawframe, background = "white", width = WIDTH, height = HEIGHT)
        drawarea.bind("<Motion>", motion)
        drawarea.bind("<ButtonPress-1>", click)
        drawarea.bind("<ButtonRelease-1>", release)
        
        # draw simultaneously on invisible image to capture later
        # necessary to save as png (otherwise need to set up ghostscript)
        # or even to get bit values as an array
        global capture, drawcapture
        capture = Image.new("RGB", (WIDTH, HEIGHT), (255,255,255))
        drawcapture = ImageDraw.Draw(capture)
        
        drawarea.pack(padx = 20, pady = 20)
        
        clearbutton=Button(drawframe, foreground = "blue", text = "Clear", command=lambda:clear(drawarea))
        clearbutton.pack(side = LEFT, expand = True, padx = 10, pady = 10)
        savebutton=Button(drawframe, foreground = "blue", text = "Save", command=lambda:save(drawarea))
        savebutton.pack(side = RIGHT, expand = True, padx = 10, pady = 10)
        
        drawframe.pack(side = RIGHT, expand = True, fill = BOTH)
        
        
'''
#Just run once:
for i in range(8):
    with Image.open(f"symbol_{i}.PNG") as im:
        resized = im.resize((WIDTH,HEIGHT))
        resized.save(f"symbol_{i}.PNG")
'''

root = Tk()
root.title("Drawing window")
window = Window(root)
root.mainloop()