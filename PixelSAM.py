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
    mask_images = []

    def __init__(self, container):
        super().__init__(container)
        # self['text'] = 'Options'
        
        # The left frame containing buttons
        self.left_tab = ttk.Frame(self)

        # Load button
        self.load_btn = tk.Button(self.left_tab, text="Load Dataset", font='sans 10 bold', height=2, width=12, background="#343434", foreground="white",  command=self.load_data)
        self.load_btn.pack(side=tk.TOP, padx=(30,30), pady=20, anchor="n")

        self.boundingbox_var = tk.IntVar()
        self.boundingbox_logo = ImageTk.PhotoImage(Image.open(os.path.join(".","assets","bbox.png")).resize((40,40), Image.Resampling.LANCZOS))
        self.boundingbox_selector = tk.Checkbutton(self.left_tab, image=self.boundingbox_logo, variable=self.boundingbox_var, font='sans 10 bold', indicatoron=False, text="Show Bbox", width=100, height = 60, compound="top", selectcolor="#34b233", command=self.statechange_callback)
        self.boundingbox_selector.pack(side=tk.BOTTOM, padx=(30,30), pady=20, anchor="n")

        # Checkbox for selecting between outer edged and all points
        # The coco based annotations have only outer edge marked
        # To make it compatible this is being added here.
        self.checkbox_var = tk.IntVar()
        self.polygon_logo = ImageTk.PhotoImage(Image.open(os.path.join(".","assets","polygon.png")).resize((40,40), Image.Resampling.LANCZOS))
        self.checkbox = tk.Checkbutton(self.left_tab, image=self.polygon_logo, variable=self.checkbox_var, font='sans 10 bold', indicatoron=False, text="Outer Edge", width=100, height = 60, compound="top", selectcolor="#34b233", command=self.statechange_callback)
        self.checkbox.pack(side=tk.BOTTOM, padx=(30,30), pady=20, anchor="n")

        self.left_tab.pack(side=tk.LEFT, padx=5, pady=5, anchor="n")

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
        
        # New object button
        self.new_btn = tk.Button(self.side_tab, text="New object", font='sans 10 bold', height=2, width=12, background="#343434", foreground="white", command = self.new_object)
        self.new_btn.pack(side=tk.BOTTOM,expand=1, padx=[10,0], pady=[10,10])

        # List of objects
        self.scrollable_list = ttk.Frame(self.side_tab)
        self.scrollable_label = ttk.Label(self.scrollable_list, text="Object Labels", font='sans 10 bold')
        self.scrollable_label.pack(side=tk.TOP,expand=1, padx=[10,0], pady=[10,0], anchor="w")
        self.yScroll = tk.Scrollbar(self.scrollable_list, orient=tk.VERTICAL)
        self.yScroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.object_list = tk.Listbox(self.scrollable_list, height=10, width=20, background="white", foreground="#343434", activestyle='none')
        self.object_list.pack(side=tk.RIGHT,expand=1, padx=[10,0], pady=[0,10])
        self.object_list.config(yscrollcommand=self.yScroll.set)
        self.yScroll.config(command=self.object_list.yview)
        # self.object_list.bind('<<ListboxSelect>>', self.onselect)
        
        # Read the data/predefined_objects.txt file and add the objects to the list
        with open(args.class_file, 'r') as f:
            for line in f:
                self.object_list.insert('end', line.strip())
            

        self.scrollable_list.pack(side=tk.BOTTOM,expand=1, padx=[10,0], pady=[10,10])
        
        # The logos
        # The logo is created using the icons from https://www.flaticon.com/free-icons/schedule and https://www.flaticon.com/free-icons/professions-and-jobs
        #PixelSAM_logo = ImageTk.PhotoImage(Image.open(os.path.join(".","assets","PixelMe.png")).resize((100,82), Image.Resampling.LANCZOS))
        #PixelSAM_logo_label = ttk.Label(self.side_tab, image=PixelSAM_logo)
        #PixelSAM_logo_label.image = PixelSAM_logo
        #PixelSAM_logo_label.pack(side=tk.BOTTOM,expand=1, padx=[10,0], pady=[0,20])

        
        self.side_tab.pack(side=tk.LEFT, anchor="s")        

        # The variables corresponding to the image player
        self.file_path = "" # Delaring path of the folder with the images to be labelled
        self.cur_annotation = [] # List of all the annotations
        self.mask_images = [] # List of all the mask images
        self.bbox_list = [] # List of all the bounding boxes
        self.annotation_count = None # The number of annotations
        self.mask_count = None # The number of mask images
        self.image_update_val = 100 # The time interval between each frame update in ms
        self.cur_image_index = 0 # The index of the current image
        self.image_list = [] # The list of images to be displayed
        self.window_height = 0 # The height of the app window
        self.state_changed = False # The state of the bbox and outer edge buttons

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
        app.bind("<n>", self.new_object)
        app.bind("<Control-Shift-Z>", self.undo_object)
        app.bind("<Control-s>", self.save_annotation)
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
        self.mask_images = []
        self.bbox_list = []
    
    # Update the frame
    def frame_update(self):
        # Get the height of the app window
        self.window_height = app.winfo_height()-20
        # Display the image
        # If a new image is loaded or there is a new annotation or the window is resized, or there is a change in the state of the buttons, then update the image
        if self.cur_image_path != self.prev_image_path or len(self.cur_annotation) != self.annotation_count or self.window_height != self.prev_window_height or self.state_changed:
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
            self.mask_count = len(self.mask_images)
            self.state_changed = False

            # Get the image dimensions
            self.img_height, self.img_width, _ = self.OCV_image.shape

            # Draw the annotations
            if len(self.cur_annotation) > 0:
                self.cv2image, self.mask_image, self.bbox_corners = SAM_prediction(self.cv2image, self.cur_annotation, self.predictor, self.img_height, self.img_width, self.mask_images, self.checkbox_var.get(), self.boundingbox_var.get())
                #get SAM polygons

                
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
        print("Right pressed")
        # Ensure that there are more images to be displayed
        if len(self.image_list) > 0:
            if self.cur_image_index < len(self.image_list)-1:
                self.cur_image_index += 1
                self.cur_image_path = os.path.join(self.file_path,self.image_list[self.cur_image_index]) # Specify the path of the image to be displayed
                self.cur_annotation = [] # Reset the annotation list
                self.mask_images = [] # Reset the mask image list
                self.bbox_list = [] # Reset the bounding box list
    
    # When the left arrow key is pressed: Display the previous image
    def left_arrow_press(self, event):
        print("Left pressed")
        # Ensure that there are more images to be displayed
        if len(self.image_list) > 0:
            if self.cur_image_index > 0:
                self.cur_image_index -= 1
                self.cur_image_path = os.path.join(self.file_path,self.image_list[self.cur_image_index]) # Specify the path of the image to be displayed
                self.cur_annotation = [] # Reset the annotation list
                self.mask_images = [] # Reset the mask image list
                self.bbox_list = [] # Reset the bounding box list
    
    # When the left mouse button is pressed: 
    def left_key_press(self, event):
        # Append the button press location to the annotation list
        if self.resize_type == "height":
            # Ensure that the click is within the image and not in the border
            if event.y > (self.window_height - self.resized_image.size[1])/2 and event.y < self.window_height-(self.window_height - self.resized_image.size[1])/2:
                self.cur_annotation.append([int(event.x*self.img_width/self.window_height), int((event.y-self.diff_dim)*self.img_height/self.resized_image.size[1]),(0, 255, 0), 1])
        else:
            # Ensure that the click is within the image and not in the border
            if event.x > (self.window_height - self.resized_image.size[0])/2 and event.x < self.window_height-(self.window_height - self.resized_image.size[0])/2:
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
        self.mask_images = []
        self.bbox_list = []

    # When a new object is to be labelled
    def new_object(self, event=None):
        if len(self.cur_annotation) > 0:
            if len(self.object_list.curselection())>0:
                self.cur_annotation = []
                self.mask_images.append(self.mask_image)
                self.bbox_list.append([self.object_list.curselection()[0], *self.bbox_corners])
                self.object_list.selection_clear(0, tk.END)
                print("Previous masks:",len(self.mask_images))
            else:
                messagebox.showwarning("Warning","Please select an object from the list")
    
    # When the undo button for the object is pressed
    def undo_object(self, event):
        if len(self.mask_images) > 0:
            self.mask_images.pop()
            self.bbox_list.pop()
            print("Previous masks:",len(self.mask_images))
    
    # When the undo button is pressed
    def undo_annotation(self, event):
        if len(self.cur_annotation) > 0:
            self.cur_annotation.pop()
    
    # When the save button is pressed
    def save_annotation(self, event):
        # If there are multiple objects labelled
        if len(self.mask_images) > 0:
            # If there is an active annotation but no label selected
            if len(self.cur_annotation) > 0 and len(self.object_list.curselection())==0:
                messagebox.showwarning("Warning","Please select an object from the list before saving")
            else:
                # If there is an active annotation
                if len(self.cur_annotation) > 0:
                    self.cur_annotation = []
                    self.mask_images.append(self.mask_image)
                    self.bbox_list.append([self.object_list.curselection()[0], *self.bbox_corners])
                    self.object_list.selection_clear(0, tk.END)
                # Save the bbox list to the file
                with open(os.path.join(self.file_path,os.path.basename(self.cur_image_path).split(".")[0]+".txt"), "w") as f:
                    for bbox in self.bbox_list:
                        f.write(str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3])+" "+str(bbox[4])+"\n")      
        # If there is only one object labelled
        elif len(self.cur_annotation) > 0:
            if len(self.object_list.curselection())>0:
                # Save the bbox list to the file
                with open(os.path.join(self.file_path,os.path.basename(self.cur_image_path).split(".")[0]+".txt"), "w") as f:
                    f.write(str(self.object_list.curselection()[0])+" "+str(self.bbox_corners[0])+" "+str(self.bbox_corners[1])+" "+str(self.bbox_corners[2])+" "+str(self.bbox_corners[3])+"\n")
            else:
                messagebox.showwarning("Warning","Please select an object from the list before saving")
        else:
            messagebox.showwarning("Warning","Please label the image before saving")
    
    # When there is a state change
    def statechange_callback(self, event=None):
        self.state_changed = True
            

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
    parser.add_argument('--class_file', type=str, default=os.path.join(os.path.dirname(__file__), "data", "predefined_classes.txt"), help='The path to the class file')

    args = parser.parse_args()
    app = App()
    controllerframe = ControlFrame(app)
    app.mainloop()
