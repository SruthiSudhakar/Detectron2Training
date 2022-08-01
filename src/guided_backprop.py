import torch
from torch.nn import ReLU
import pdb
import cv2

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.target_layer_name = target_layer_name
        self.handlers = []  # a set of hook function handlers
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        print('registering hooks')
        def hook_function(module, grad_in, grad_out):
            print('HOOOK FUNCTION', grad_in[0].shape, grad_out[0].shape)
            self.gradients = grad_in[0]
        pdb.set_trace()
        first_layer = list(list(self.model._modules.items())[0][1]._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)
        # for (name, module) in self.model.named_modules():
        #     if name == self.target_layer_name:
        #         module.register_backward_hook(hook_function)
        #         return
        print(f"Layer {self.target_layer_name} not found in Model!")
  
    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for (name, module) in self.model.named_modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, inputs, target_category):
        # Forward pass
        output = self.model.forward([inputs])
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        # one_hot_output = torch.FloatTensor(1, len(output[0]['instances'].scores_for_all_classes)).zero_()
        # one_hot_output[0][target_category] = 1
        # Backward pass
        if len(output[0]['instances'].scores_for_all_classes) > target_category:
            score = output[0]['instances'].scores_for_all_classes[target_category]
        else:
            print('NO DETECTIONS')
            target_category = -1
            return np.zeros((inputs["height"],inputs["width"])), np.zeros((inputs["height"],inputs["width"])), output, target_category
        score.backward()
#         output[0]['instances'].scores_for_all_classes.reshape((1,21)).backward(gradient=one_hot_output.to('cuda:0'))
#         one_hot_output = one_hot_output.to('cpu')
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.cpu().data.numpy()[0].mean(axis=0)
        gradients_as_arr -= gradients_as_arr.min()
        gradients_as_arr /= gradients_as_arr.max()
        return gradients_as_arr, target_category