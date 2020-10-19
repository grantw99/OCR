from keras.models import load_model
import pandas as pd
import numpy as np
from PIL import Image,ImageOps
import CharacterSegmentation as cs
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import cv2

class Predictor:
    def __init__(self, model_dir='./model/alphanum_model_binary_ex88.h5', segmented_dir='./segmented', mapping_processed='./data/processed_mapping.csv'):
        self.model = load_model(model_dir)
        self.segmented_dir = segmented_dir
        self.mapping_processed = mapping_processed

    def img2emnist(self, filepath, char_code):
        img = Image.open(filepath).resize((28,28))
        inv_img = ImageOps.invert(img)

        flatten = np.array(inv_img).flatten()
        flatten = flatten / 255
        flatten = np.where(flatten > 0.5, 1, 0)

        csv_img = ','.join([str(num) for num in flatten])

        csv_str = '{},{}'.format(char_code, csv_img)
        return csv_str

    def predict(self, img_dir):
        cs.image_segmentation(img_dir)
        files = os.listdir(self.segmented_dir)
        files = sorted(files)
        for f in files:
            filename = self.segmented_dir + f
            img = cv2.imread(filename, 0)

            kernel = np.ones((2,2), np.uint8)
            dilation = cv2.erode(img, kernel, iterations = 1)
            cv2.imwrite(filename, dilation)
        temp_filename = 'test.csv'
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        f_test= open(temp_filename, 'a+')

        f_test.close()

        data = pd.read_csv(temp_filename)
        X_data = data.drop(labels = ["label"], axis = 1)
        X_data = X_data.values.reshape(-1,28,28,1)

        df = pd.read_csv(self.mapping_processed)
        code2char = {}
        for index, row in df.iterrows():
            code2char[row['id']] = row['char']

        results = self.model.predict(X_data)

        results = np.argmax(results, axis = 1)
        parsed_str = ""
        for r in results:
            parsed_str += code2char[r]

        return parsed_str

    
    
        



