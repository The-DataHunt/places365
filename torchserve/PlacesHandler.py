import logging
import numpy as np
from PIL import Image, ImageDraw
import requests
import json
import io
import os
import wideresnet
from torchvision import transforms as trn
import torch.nn.functional as F
from base64 import encodebytes
import torch
from ts.torch_handler.base_handler import BaseHandler

class PlacesHandler(BaseHandler):
    def __init__(self, *args, **kwargs):
        self.model = None
        self.initialized = False
        self.device = "cpu" 
        self.model_dir = ""
        self.classes = -1
        self.labels_IO = []
        self.labels_attribute = []
        self.W_attribute = None

    def initialize(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.model_dir = model_dir
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        model_file = os.path.join(model_dir, 'wideresnet18_places365.pth.tar')

        self.model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.tf = self.returnTF()
        self.classes, self.labels_IO, self.labels_attribute, self.W_attribute = self.load_labels()
        
        self.initialized = True
    
    def load_labels(self):
        # prepare all the labels
        # scene category relevant
        file_name_category = 'categories_places365.txt'
        if not os.access(file_name_category, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

        # indoor and outdoor relevant
        file_name_IO = 'IO_places365.txt'
        if not os.access(file_name_IO, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
            os.system('wget ' + synset_url)
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)

        # scene attribute relevant
        file_name_attribute = 'labels_sunattribute.txt'
        if not os.access(file_name_attribute, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
            os.system('wget ' + synset_url)
        with open(file_name_attribute) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        file_name_W = 'W_sceneattribute_wideresnet18.npy'
        if not os.access(file_name_W, os.W_OK):
            synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
            os.system('wget ' + synset_url)
        W_attribute = np.load(file_name_W)

        return classes, labels_IO, labels_attribute, W_attribute

    def returnTF(self):
        # load the image transformer
        tf = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return tf

   
    def preprocess_one_image(self, req):
        json_dict = req['body'] if 'body' in req else req
        img_url    = json_dict['imgurl'] if 'imgurl' in json_dict else ''
        image_bytes = json_dict['file'] if 'file' in json_dict else None

        if img_url:
            response = requests.get(img_url)
            org_img = Image.open(io.BytesIO(response.content))
        else :
            #image_bytes = image_bytes.read()
            org_img = Image.open(io.BytesIO(image_bytes))
        
        org_img = org_img.convert('RGB')
        img_tensor = self.tf(org_img).unsqueeze(0)
        return org_img, img_tensor

    def preprocess(self, reqs):
        processed = [self.preprocess_one_image(req) for req in reqs]
        return processed

    def inference(self, x):
        outs = []
        with torch.no_grad():
            for d in x:
                logit = self.model(d[1])
                h_x = F.softmax(logit, 1).data.squeeze()
                probs, idx = h_x.sort(0, True)
                probs = probs.numpy()
                idx = idx.numpy()
                
                io_image = np.mean(self.labels_IO[idx[:10]]) # vote for the indoor or outdoor
                io_voted = "indoor" if io_image < 0.5 else "outdoor"
                cur_dict = {
                    "top5" : [(self.classes[idx[i]], str(probs[i]))for i in range(5)],
                    "env"  : io_voted
                }
                outs.append([d[0], cur_dict])

        return outs

    def postprocess(self, preds):
        output = []
        for img, pred in preds:
            # each pred is a 2d-array containing the list of coords
            scene_list = json.dumps(pred, indent=2, ensure_ascii=False)

            # save the image into byte array
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='JPEG')
            img = encodebytes(byte_arr.getvalue()).decode('ascii')

            output.append({"detections" : scene_list, "image": img})
        return output

_service = PlacesHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None
    
    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data