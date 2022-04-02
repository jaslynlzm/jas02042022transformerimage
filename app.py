#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image #use PIL
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = "static/"


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        print("File Received")
        filename = secure_filename(file.filename)
        file.save(filename)
        file = open(filename,"r")
        image = Image.open(filename)
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        #model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        pred = model.config.id2label[predicted_class_idx]
        #pred = "1"
        return(render_template("index.html", result=str(pred)))
    else:
        return(render_template("index.html", result="2"))
    
if __name__ == "__main__":
    app.run()


# In[ ]:





# In[ ]:




