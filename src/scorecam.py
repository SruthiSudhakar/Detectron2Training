"""
Created on Wed Apr 29 16:11:20 2020

@author: Haofan Wang - github.com/haofanwang
"""
from PIL import Image
import numpy as np
import torch, pdb
import torch.nn.functional as F

from misc_functions import get_example_params, save_class_activation_images

class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradient = None
        self.model.eval()
        self.activations_grads = []
        self._register_hook()

    def _get_activations_hook(self, module, input, output):
        # pdb.set_trace()
        self.conv_output = output
    
    def _register_hook(self):
        print('registering hooks')
        for (name, module) in self.model.named_modules():
            if name == self.target_layer:
                # pdb.set_trace()
                self.activations_grads.append(module.register_forward_hook(self._get_activations_hook))
                return
        print(f"Layer {self.target_layer} not found in Model!")

    def _release_activations_grads(self):
        for handle in self.activations_grads:
            handle.remove()
    

    def generate_cam(self, inputs):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        _ = self.model([inputs])
        # Get convolution outputs
        target = self.conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(inputs['image'].shape[1], inputs['image'].shape[2]), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            pdb.set_trace()
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            inputs['image'] = inputs['image'].to("cuda:0")
            inputs['image'] *= norm_saliency_map[0]
            w = F.softmax(self.model([inputs])[1],dim=1)[0][0]
            cam += w.data.numpy() * target[i, :, :].data.numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((inputs['image'].shape[2],
                       inputs['image'].shape[3]), Image.ANTIALIAS))/255
        return cam
