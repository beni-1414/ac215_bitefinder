from model import *

model_classes = {
    'clip': CLIPForBugBiteClassification,
    'vilt': ViLTForBugBiteClassification,
}

def model_class(model):
    return next((model_class_id for model_class_id, model_class in model_classes.items() if isinstance(model, model_class)), None)