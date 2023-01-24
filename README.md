
# Crop YOLO Bounding boxes 

The script will crop the bounding box of YOLO models such as YOLOv4, YOLOv5, YOLOv7, and YOLOv8. YOLO annotations are normalized so it is tricky to crop the annotation if you have not done it before. The annotations have to be converted to unnormalized format to crop the label in an image. A sample label for a person is given as:
```python
0 0.2344 0.7833 0.0343 0.89838
```
This is for just demonstration, it's not a true label. This is how YOLO format annotations look like. The first number (integer) is class label such as person in the example above. Second number is called "x" and it's bounding box's center
coordinate, similary, third number (y) is also a center coordinate. And, rest two numbers are width (w) and height (h) of the bouding box. 

This will transform the annotations to xyxy (unnormalized top left point and bottom right point) of the bouding box for cropping the label in an image.




## Installation

To install the repository, run following commands.

```bash
https://github.com/faizan1234567/Crop-YOLO-Bounding-Boxes.git
cd Crop-YOLO-Bounding-Boxes
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
    
## Usage
```python
python crop_yolo_labels.py --images path_to_images_directory --save path_to_directory_to_save_crop_images --labels path_to_labels_directory
```
I will add more featuer to the repository like cropping yolo predictions etc. If it's useful for you, please star the repository. 
