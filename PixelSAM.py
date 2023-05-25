import argparse
import cv2 
import numpy as np
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog,messagebox,ttk
from segment_anything import sam_model_registry, SamPredictor
from utils.general import select_device
from utils.predict import SAM_setup, SAM_prediction
import torch
import time
import threading
import webbrowser
        
# Class for the main window
class ControlFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        # self['text'] = 'Options'
        
        # Load button
        self.load_btn = tk.Button(self, text="Load Dataset", font='sans 10 bold', height=2, width=12, background="#343434", foreground="white",  command=self.load_data)
        self.load_btn.pack(side=tk.LEFT, padx=(30,30), pady=20, anchor="n")

        # The frame which includes the image player
        self.imageplayer = ttk.Frame(self)

        # The Image Tkinter label
        self.imagelabel = ttk.Label(self.imageplayer)
        self.imagelabel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        self.imageplayer.pack(side=tk.LEFT, padx=5, pady=5, expand=1, fill=tk.BOTH)

        # The side frame which include the help button and the reset
        # NOTE: There may be more things supported in the future
        self.side_tab = ttk.Frame(self)
        
        # Help button
        self.help_btn = tk.Button(self.side_tab, text="Help", font='sans 10 bold', height=2, width=12, background="#343434", foreground="white", command = self.help_btn_browser)
        self.help_btn.pack(side=tk.BOTTOM,expand=1, padx=[10,0], pady=[10,50])
        
        # Reset button
        self.reset_btn = tk.Button(self.side_tab, text="Reset image", font='sans 10 bold', height=2, width=12, background="#343434", foreground="white", command = self.reset_annotation)
        self.reset_btn.pack(side=tk.BOTTOM,expand=1, padx=[10,0], pady=[10,10])
        
        # The logos
        # The logo is created using the icons from https://www.flaticon.com/free-icons/schedule and https://www.flaticon.com/free-icons/professions-and-jobs
        PixelSAM_logo = ImageTk.PhotoImage(Image.open(os.path.join(".","assets","PixelMe.png")).resize((100,82), Image.Resampling.LANCZOS))
        PixelSAM_logo_label = ttk.Label(self.side_tab, image=PixelSAM_logo)
        PixelSAM_logo_label.image = PixelSAM_logo
        PixelSAM_logo_label.pack(side=tk.BOTTOM,expand=1, padx=[10,0], pady=[0,20])

        
        self.side_tab.pack(side=tk.LEFT, anchor="s")        

        # The variables corresponding to the image player
        self.file_path = "" # Delaring path of the folder with the images to be labelled
        self.cur_annotation = [] # List of all the annotations
        self.annotation_count = None # The number of annotations
        self.image_update_val = 100 # The time interval between each frame update in ms
        self.cur_image_index = 0 # The index of the current image
        self.image_list = [] # The list of images to be displayed
        self.window_height = 0 # The height of the app window

        # Path to the intro image
        self.cur_image_path = os.path.join(".","assets","intro.png")
        self.prev_image_path = ""

        # Load the model for segmentation
        if args.model_path == "sam_vit_h_4b8939.pth":
            self.predictor = SAM_setup(args.model_type, os.path.join(".","sam_vit_h_4b8939.pth"), select_device(""))
        else:
            self.predictor = SAM_setup(args.model_type, args.model_path, select_device(""))
        
        # The function responsible for updating the frame
        self.frame_update()

        # Binding the keys and mouse clicks
        app.bind("<Right>", self.right_arrow_press)
        app.bind("<Left>", self.left_arrow_press)
        app.bind("<Control-z>", self.undo_annotation)
        self.imagelabel.bind('<1>', self.left_key_press)
        self.imagelabel.bind('<3>', self.right_key_press)

        self.grid(column=0, row=0, padx=5, pady=5, sticky='ew')
    
    # Function to load the data
    def load_data(self):
        self.file_path = filedialog.askdirectory()
        if self.file_path:
            if len([x for x in os.listdir(self.file_path) if x.endswith(".jpg") or x.endswith(".png")]) > 0:
                self.set_image(self.file_path)
            else:
                print("No images found in the selected directory")
    
    # Setting the images to be displayed
    def set_image(self, file_path):
        self.file_path=file_path
        # list all the jpg and png images in the folder
        self.image_list = os.listdir(self.file_path)
        self.image_list = [x for x in self.image_list if x.endswith(".jpg") or x.endswith(".png")]

        self.cur_image_path = os.path.join(self.file_path,self.image_list[0])
        self.cur_image_index = 0
        self.cur_annotation = []
    
    # Update the frame
    def frame_update(self):
        # Get the height of the app window
        self.window_height = app.winfo_height()-20
        # Display the image
        if self.cur_image_path != self.prev_image_path or len(self.cur_annotation) != self.annotation_count or self.window_height != self.prev_window_height:
            # Read the image and convert it to RGB
            self.OCV_image = cv2.imread(self.cur_image_path)
            self.cv2image = cv2.cvtColor(self.OCV_image, cv2.COLOR_BGR2RGB)
            # Only when a new image is selected, the predictor image is set
            if self.cur_image_path != self.prev_image_path and self.prev_image_path != "":
                # Show a loading message box
                print("Loading the image. Please wait...")
                self.predictor.set_image(self.cv2image)
                # Close the loading message box
                print("Image loaded")
            
            # Update the previous image path, the annotation count and the window height
            self.prev_image_path = self.cur_image_path
            self.annotation_count = len(self.cur_annotation)
            self.prev_window_height = self.window_height

            # Get the image dimensions
            self.img_height, self.img_width, _ = self.OCV_image.shape

            # Draw the annotations
            if len(self.cur_annotation) > 0:
                self.cv2image = SAM_prediction(self.cv2image, self.cur_annotation, self.predictor, self.img_height, self.img_width)
                for i in range(len(self.cur_annotation)):
                    self.cv2image = cv2.circle(self.cv2image, (self.cur_annotation[i][0], self.cur_annotation[i][1]), int((self.img_height+self.img_width)/200), self.cur_annotation[i][2], -1)
            
            img = Image.fromarray(self.cv2image)

            # Resize the image to fit the window
            if int(self.img_width*self.window_height/self.img_height) > self.window_height:
                self.resized_image = img.resize((self.window_height, int(self.img_height*self.window_height/self.img_width)), Image.Resampling.LANCZOS)
                self.resize_type = "height"
                self.diff_dim = (self.window_height - self.resized_image.size[1])/2 # The difference between the image height and the window height
            else:
                self.resized_image = img.resize((int(self.img_width*self.window_height/self.img_height),self.window_height), Image.Resampling.LANCZOS)
                self.resize_type = "width"
                self.diff_dim = (self.window_height - self.resized_image.size[0])/2 # The difference between the image width and the window width
            
            # Add black bars to the image
            image_new = Image.new("RGB", (self.window_height, self.window_height), (0, 0, 0))
            # Place image at the center of this new image
            image_new.paste(self.resized_image, (int((self.window_height - self.resized_image.size[0]) / 2), int((self.window_height - self.resized_image.size[1]) / 2)))
            
            # Convert the image to ImageTk format
            imgtk = ImageTk.PhotoImage(image_new)
            self.imagelabel.imgtk = imgtk
            self.imagelabel.configure(image=imgtk)
        # Update the frame after the specified time interval
        self.imageplayer.after(self.image_update_val, self.frame_update)

    
    # When the right arrow key is pressed: Display the next image
    def right_arrow_press(self, event):
        # Ensure that there are more images to be displayed
        if len(self.image_list) > 0:
            if self.cur_image_index < len(self.image_list)-1:
                self.cur_image_index += 1
                self.cur_image_path = os.path.join(self.file_path,self.image_list[self.cur_image_index])
                self.cur_annotation = []
    
    # When the left arrow key is pressed: Display the previous image
    def left_arrow_press(self, event):
        # Ensure that there are more images to be displayed
        if len(self.image_list) > 0:
            if self.cur_image_index > 0:
                self.cur_image_index -= 1
                self.cur_image_path = os.path.join(self.file_path,self.image_list[self.cur_image_index])
                self.cur_annotation = []
    
    # When the left mouse button is pressed: 
    def left_key_press(self, event):
        # Append the button press location to the annotation list
        if self.resize_type == "height":
            # Ensure that the click is within the image and not in the border
            if event.y > (self.window_height - self.resized_image.size[1])/2 or event.y < self.window_height-(self.window_height - self.resized_image.size[1])/2:
                self.cur_annotation.append([int(event.x*self.img_width/self.window_height), int((event.y-self.diff_dim)*self.img_height/self.resized_image.size[1]),(0, 255, 0), 1])
        else:
            # Ensure that the click is within the image and not in the border
            if event.x > (self.window_height - self.resized_image.size[0])/2 or event.x < self.window_height-(self.window_height - self.resized_image.size[0])/2:
                self.cur_annotation.append([int((event.x-self.diff_dim)*self.img_width/self.resized_image.size[0]), int(event.y*self.img_height/self.window_height),(0, 255, 0), 1])
    
    # When the left mouse button is pressed: 
    def right_key_press(self, event):
        # Append the button press location to the annotation list
        if self.resize_type == "height":
            # Ensure that the click is within the image and not in the border
            if event.y > (self.window_height - self.resized_image.size[1])/2 or event.y < self.window_height-(self.window_height - self.resized_image.size[1])/2:
                self.cur_annotation.append([int(event.x*self.img_width/self.window_height), int((event.y-self.diff_dim)*self.img_height/self.resized_image.size[1]),(255, 0, 0), 0])
        else:
            # Ensure that the click is within the image and not in the border
            if event.x > (self.window_height - self.resized_image.size[0])/2 or event.x < self.window_height-(self.window_height - self.resized_image.size[0])/2:
                self.cur_annotation.append([int((event.x-self.diff_dim)*self.img_width/self.resized_image.size[0]), int(event.y*self.img_height/self.window_height),(255, 0, 0), 0])

    # When the help button is pressed
    def help_btn_browser(self):
        # webbrowser.open(r"https://www.google.com",autoraise=True)
        pass

    # When the reset button is pressed
    def reset_annotation(self):
        self.cur_annotation = []
    
    # When the undo button is pressed
    def undo_annotation(self, event):
        if len(self.cur_annotation) > 0:
            self.cur_annotation.pop()

# App class
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PixelSAM Labelling Tool")
        self.geometry("1120x650+10+10")
        self.minsize(1120,650)
        # self.resizable(False, False)
        # The logo is created using the icons from https://www.flaticon.com/free-icons/schedule and https://www.flaticon.com/free-icons/professions-and-jobs
        self.PixelSAM_icon = ImageTk.PhotoImage(Image.open(os.path.join(".","assets","PixelMe.png")).resize((80,80), Image.Resampling.LANCZOS))
        self.iconphoto(False, self.PixelSAM_icon)
        self.protocol("WM_DELETE_WINDOW",self.on_closing)

    
    # Function to close the app
    def on_closing(self):
        if messagebox.askokcancel("Quit","Are you sure?"):
            #exit_event.set()
            self.quit()
            self.destroy()
            os._exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PixelSAM Labelling Tool supports image and video labelling for object classification\
                        Example: python PixelSAM.py') 
    parser.add_argument('--model_path', type=str, default="sam_vit_h_4b8939.pth", help='The path to the model checkpoint')
    parser.add_argument('--model_type', type=str, default="vit_h", help='The path to the model checkpoint')

    args = parser.parse_args()
    app = App()
    controllerframe = ControlFrame(app)
    app.mainloop()
