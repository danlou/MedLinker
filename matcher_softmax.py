import joblib
from keras.models import load_model

import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(42)

class SoftMax_CLF():

    def __init__(self, threshold=0.5):
        self.label_mapping = None
        self.model = None
        self.threshold = threshold

    def load(self, model_path, mapping_path):
        self.model = load_model(model_path)
        self.label_mapping = joblib.load(mapping_path)
        self.label_mapping = {i:l for l, i in self.label_mapping.items()}

    def predict(self, ctx_vec):
        preds = self.model.predict_proba(ctx_vec.reshape(1, -1))[0]

        matches = {}
        for pred_idx, pred in enumerate(preds):
            if pred > self.threshold:  # skips NaNs too
                pred_label = self.label_mapping[pred_idx]
                pred_label = pred_label.lstrip('UMLS:')
                matches[pred_label] = pred
        
        matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)

        return matches
    