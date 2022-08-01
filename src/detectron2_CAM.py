"""
This overrides are to make detectron2 compatible to GradCAM
mainly by adding scores for all classes

- function fast_rcnn_inference_single_image() inside fast_rcnn module
- methods set() and __getitem__() in Instances Class
"""

import torch

from typing import Tuple, Any, Union
from detectron2.structures import Boxes
from detectron2.layers import batched_nms

# for overriding
from detectron2.modeling.roi_heads import fast_rcnn
from detectron2.structures import Instances

def fast_rcnn_inference_single_image(
        boxes,
        scores,
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    indices = torch.arange(start=0, end=scores.shape[0], dtype=int)
    indices = indices.expand((scores.shape[1], scores.shape[0])).T

    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        indices = indices[valid_mask]

    scores = scores[:, :-1]

    scores_for_all_classes = scores.amax(axis=0)

    indices = indices[:, :-1]

    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    indices = indices[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    indices = indices[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.indices = indices
    result.scores_for_all_classes = scores_for_all_classes

    return result, filter_inds[:, 0], scores_for_all_classes

def set(self, name: str, value: Any) -> None:
    """
    Set the field named `name` to `value`.
    The length of `value` must be the number of instances,
    and must agree with other existing fields in this object.
    """
    if name != 'scores_for_all_classes':
        data_len = len(value)
        if len(self._fields):
            assert (
                    len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
    self._fields[name] = value

def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
    """
    Args:
        item: an index-like object and will be used to index all the fields.

    Returns:
        If `item` is a string, return the data in the corresponding field.
        Otherwise, returns an `Instances` where all fields are indexed by `item`.
    """
    if type(item) == int:
        if item >= len(self) or item < -len(self):
            raise IndexError("Instances index out of range!")
        else:
            item = slice(item, None, len(self))
    ret = Instances(self._image_size)
    for k, v in self._fields.items():
        if k == 'scores_for_all_classes':
            ret.set(k, v)
        else:
            ret.set(k, v[item])
    return ret


fast_rcnn.fast_rcnn_inference_single_image = fast_rcnn_inference_single_image

Instances.set = set
Instances.__getitem__ = __getitem__
