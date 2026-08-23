from transformers import XCLIPProcessor, XCLIPModel, XCLIPVisionConfig, XCLIPConfig, XCLIPProcessor, XCLIPVisionModel
from transformers import VivitImageProcessor, VivitModel

def load_model(model_name, vision_component=False):

    if 'xclip' in model_name:
        processor = XCLIPProcessor.from_pretrained(model_name)
        if vision_component: # used for detailed output for getting spatial features
            config = XCLIPVisionConfig(output_hidden_states=True, return_dict=True)
            model = XCLIPVisionModel(config=config).from_pretrained(model_name)
        else:
            model = XCLIPModel.from_pretrained(model_name)
        
    elif 'vivit' in model_name:
        processor = VivitImageProcessor.from_pretrained(model_name)
        model = VivitModel.from_pretrained(model_name)

    else:
        processor, model = None, None
    
    return processor, model