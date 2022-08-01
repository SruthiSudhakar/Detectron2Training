import cv2
import numpy as np
import pdb
class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, model, target_layer_name):

        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradient = None
        self.model.eval()
        self.activations_grads = []
        self._register_hook()

    def _get_activations_hook(self, module, input, output):
        self.activations = output
    
    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hook(self):
        print('registering hooks')
        for (name, module) in self.model.named_modules():
            if name == self.target_layer_name:
                self.activations_grads.append(module.register_forward_hook(self._get_activations_hook))
                self.activations_grads.append(module.register_backward_hook(self._get_grads_hook))
                return
        print(f"Layer {self.target_layer_name} not found in Model!")

    def _release_activations_grads(self):
        for handle in self.activations_grads:
            handle.remove()
    
    def _postprocess_cam(self, raw_cam, img_width, img_height):
        cam_orig = np.sum(raw_cam, axis=0)  # [H,W]
        cam_orig = np.maximum(cam_orig, 0)  # ReLU
        cam_orig -= np.min(cam_orig)
        cam_orig /= np.max(cam_orig)
        cam = cv2.resize(cam_orig, (img_width, img_height))
        return cam, cam_orig
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._release_activations_grads()

    def __call__(self, inputs, target_category=0, type='tsne'):
        """
        Calls the GradCAM++ instance

        Parameters
        ----------
        inputs : dict
            The input in the standard detectron2 model input format
            https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format

        target_category : int, optional
            The target category index. If `None` the highest scoring class will be selected

        Returns
        -------
        cam : np.array()
          Gradient weighted class activation map
        output : list
          list of Instance objects representing the detectron2 model output
        """
        self.model.zero_grad()
        output = self.model.forward([inputs])
        if type =='tsne':
            class_id = output[0]['instances'].pred_classes[0].detach().cpu().data.numpy()
            proposal_idx = output[0]['instances'].indices[0]
            feature = self.activations[proposal_idx].detach().cpu().data  # [C,H,W]
            return feature, class_id

        # if target_category == None:
        #     target_category =  np.argmax(output[0]['instances'].scores.cpu().data.numpy(), axis=-1)
        # target_category_idx = np.argwhere(output[0]['instances'].pred_classes.detach().cpu().data.numpy() == target_category)
        if len(output[0]['instances'].scores_for_all_classes) > target_category:
            score = output[0]['instances'].scores_for_all_classes[target_category]
        else:
            print('NO DETECTIONS')
            target_category = -1
            return np.zeros((inputs["height"],inputs["width"])), np.zeros((inputs["height"],inputs["width"])), output, target_category
        # except:
        #     try: 
        #         score = output[0]['instances'].scores[0]
        #         target_category = output[0]['instances'].pred_classes[0].detach().cpu().data.numpy()
        #     except:
        #         return np.zeros((inputs["height"],inputs["width"])), np.zeros((inputs["height"],inputs["width"])), output, target_category
        #box0 = output[0]['instances'].pred_boxes[0].tensor[0][target_category]
        #print(box0)
        #box0.backward()
        score.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        activations = self.activations[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam, cam_orig = self._postprocess_cam(cam, inputs["width"], inputs["height"])
        return cam, cam_orig, output, target_category