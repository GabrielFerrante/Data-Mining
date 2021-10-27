"""
Categorias do YOLOv3
[ “person”, “bicycle”, “car”, “motorcycle”, “airplane”, “bus”, “train”,
 “truck”, “boat”, “traffic light”, “fire hydrant”, “stop sign”, “parking meter”,
 “bench”, “bird”, “cat”, “dog”, “horse”, “sheep”, “cow”, “elephant”, “bear”,
 “zebra”, “giraffe”, “backpack”, “umbrella”, “handbag”, “tie”, “suitcase”, 
 “frisbee”, “skis”, “snowboard”, “sports ball”, “kite”, “baseball bat”, 
 “baseball glove”, “skateboard”, “surfboard”, “tennis racket”, “bottle”, 
 “wine glass”, “cup”, “fork”, “knife”, “spoon”, “bowl”, “banana”, “apple”, 
 “sandwich”, “orange”, “broccoli”, “carrot”, “hot dog”, “pizza”, “donut”, 
 “cake”, “chair”, “couch”, “potted plant”, “bed”, “dining table”, “toilet”, 
 “tv”, “laptop”, “mouse”, “remote”, “keyboard”, “cell phone”, “microwave”, 
 “oven”, “toaster”, “sink”, “refrigerator”, “book”, “clock”, “vase”, “scissors”, 
 “teddy bear”, “hair drier”, “toothbrush”]

"""

#INSTALANDO NO GOOGLE COLAB

"""
!git clone https://github.com/ultralytics/yolov3  # clone repo
%cd yolov3
%pip install -qr requirements.txt  # install dependencies

"""

import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")



"""
Detecção de objetos
Na primeira execução, é necessário baixar o modelo.

"""

#!python detect.py --weights yolov3.pt --img 640 --conf 0.25 --source data/images/

Image(filename='runs/detect/exp/zidane.jpg', width=600)

Image(filename='runs/detect/exp/bus.jpg', width=600)

#TESTANDO PARA OUTRAS IMAGENS

#!wget -O data/images/microwave.jpg https://media-cdn.tripadvisor.com/media/photo-s/08/41/fd/2f/old-chair-microwave-on.jpg

Image(filename='runs/detect/exp2/microwave.jpg', width=600)

#!wget -O data/images/protesto.jpg https://www.cnnbrasil.com.br/wp-content/uploads/sites/12/2021/09/FUP20210829167.jpg

Image(filename='runs/detect/exp3/protesto.jpg', width=600)

