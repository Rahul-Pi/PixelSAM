# PixelSAM
PixelSAM is a is a graphical image annotation tool powered by the [Segment Anything Model](https://github.com/facebookresearch/segment-anything)

The tool allows users to segment the images automatically making the labelling process more smooth.

It is based on Python and uses Tkinter for the UI.

# Getting started
Install Segment Anything:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Clone the PixelSAM Repository:
```
git clone git@github.com:Rahul-Pi/PixelSAM.git
```
Download the Model Checkpoints:
```
cd PixelSAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
Note: If the link does not work or you want to choose a different model go the [segment anything repository](https://github.com/facebookresearch/segment-anything#model-checkpoints) and download.

To Run:
```
python3 PixelSAM.py
```

Click on the load button and select the folder containing the images.

## Keys and actions
Key/button | Actions
--- | ---
→ | Go to next image
← | Go to previous image
Left mouse click | Add areas to segment
Right mouse click | Exclude areas from segment

## Command line arguments
```
  --model_path: path to model if not in the same folder as the script
    (default='sam_vit_h_4b8939.pth')
  --model_type: type of the model checkpoint used
    (default='vit_h')
```

### How to contribute
Send a pull request

### TODO
- [ ] Save the labels
- [ ] Label multiple objects
- [ ] Improve the polygon detection
- [ ] Possibility to zoom
- [ ] Drawing bounding box
