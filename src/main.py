"""Script for Detectron2 training, eval, and prediction on Azure ML."""
# TODO: Implement mid-epoch saving and resuming for preemptible nodes.
import os.path as osp
import numpy as np
import pandas as pd

# Imports Detectron2 libraries.
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.engine import default_setup, DefaultPredictor, DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image
from detectron2.structures import Boxes, pairwise_iou
from detectron2.checkpoint import DetectionCheckpointer

from grad_cam import GradCAM
from drise import DRISE
from deep_dream import DeepDream
from scorecam import ScoreCam
from LRP import LRP
from guided_backprop import GuidedBackprop

from PIL import Image

import cv2, torch, json, os, pdb, utils, glob
import matplotlib.pyplot as plt
import misc_functions

from args import parse_args
from COCOAzure import COCOAzureTrainer, COCOAzureTransformerTrainer
from azureml.core.run import Run
from tqdm import tqdm
from detectron2.data.samplers import InferenceSampler
from d2.detr import add_detr_config

import detectron2_CAM

def set_pt_dirs(args):
    """Prepends phillytools data dir if available."""
    if "PT_DATA_DIR" in os.environ:
        pt_data_dir = os.environ["PT_DATA_DIR"]

        args.train_dirs = [osp.join(pt_data_dir, x) for x in args.train_dirs] if args.train_dirs else args.train_dirs
        args.train_jsons = [osp.join(pt_data_dir, x) for x in args.train_jsons] if args.train_jsons else args.train_jsons

        args.test_dir = [osp.join(pt_data_dir, x) for x in args.test_dir] if args.test_dir else args.test_dir
        args.test_json = [osp.join(pt_data_dir, x) for x in args.test_json] if args.test_json else args.test_json

        args.weights = osp.join(pt_data_dir, args.weights) if args.weights else args.weights

        args.output_dir = osp.join(pt_data_dir, args.output_dir)
    return args

def register_dataset(name, data_json, data_dir):
    """Registers COCO-style dataset into Detectron2."""

    register_coco_instances(name, {}, data_json, data_dir)

    return DatasetCatalog.get(name), MetadataCatalog.get(name)

def load_cfg(args):
    """Merges configs with Detectron2 defaults."""

    cfg = get_cfg()
    
    if args.transformer:
        add_detr_config(cfg)
        base_cfg = os.path.join( os.path.abspath(os.getcwd()), 'src/d2/configs/detr_256_6_6_torchvision.yaml')
        cfg.merge_from_file(base_cfg)
        cfg.MODEL.DETR.NUM_CLASSES = args.classes
        cfg.MODEL.WEIGHTS = args.weights if args.weights is not None else 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
    else: 
    
        # Gets and loads base config from model zoo.
        base_cfg = model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.merge_from_file(base_cfg)
        cfg.MODEL.WEIGHTS = args.weights if args.weights is not None else model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.proposals if args.proposals is not None else 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.classes

        # Sets test threshold for prediction only.
        if not args.train:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
        
    # Sets implementation parameters.
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.SIZE = [0.5, 0.5]
    cfg.INPUT.CROP.TYPE = "relative_range" 
    
    cfg.DATALOADER.NUM_WORKERS = args.workers if args.workers is not None else cfg.DATALOADER.NUM_WORKERS
    cfg.OUTPUT_DIR = args.output_dir

    # Sets solver parameters.
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.GAMMA = args.gamma if args.gamma is not None else cfg.SOLVER.GAMMA
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size if args.batch_size is not None else cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.LR_SCHEDULER_NAME = args.scheduler if args.scheduler is not None else cfg.SOLVER.LR_SCHEDULER_NAME
    cfg.SOLVER.MAX_ITER = args.iters if args.iters is not None else cfg.SOLVER.MAX_ITER
    cfg.SOLVER.STEPS = args.steps if args.steps is not None else cfg.SOLVER.STEPS if args.gamma is not None else cfg.SOLVER.STEPS
    cfg.SOLVER.WARMUP_FACTOR = args.warmup_factor if args.warmup_factor is not None else cfg.SOLVER.WARMUP_FACTOR
    cfg.SOLVER.WARMUP_ITERS = args.warmup_iters if args.warmup_iters is not None else cfg.SOLVER.WARMUP_ITERS
    cfg.SOLVER.WARMUP_METHOD = args.warmup_method if args.warmup_method is not None else cfg.SOLVER.WARMUP_METHOD
    cfg.SOLVER.CHECKPOINT_PERIOD = int(cfg.SOLVER.MAX_ITER/5)

    # Sets datasets and directories.
    cfg.DATASETS.TRAIN = tuple([x.replace("/","_")+'_train' for x in args.train_dirs]) if args.train_dirs else ()
    if args.test_dir:
        cfg.DATASETS.TEST = tuple([x.replace("/","_")+'_val' for x in args.test_dir]) if args.test_dir else ()
        cfg.TEST.EVAL_PERIOD = int(cfg.SOLVER.MAX_ITER/5)
    else:
        cfg.DATASETS.TEST = ()
    cfg.freeze()
    return cfg

def coco_bbox_to_coordinates(bbox):
    out = bbox.copy().astype(float)
    out[:, 2] = bbox[:, 0] + bbox[:, 2]
    out[:, 3] = bbox[:, 1] + bbox[:, 3]
    return out

def conf_matrix_calc(image_name, labels, detections, n_classes, conf_thresh, iou_thresh):
    confusion_matrix = np.zeros([n_classes + 1, n_classes + 1])
    
    if len(labels) == 0:
        return confusion_matrix
    l_classes = np.array(labels)[:, 0].astype(int)
    l_bboxs = coco_bbox_to_coordinates((np.array(labels)[:, 1:]))
    d_confs = np.array(detections)[:, 4]
    d_bboxs = (np.array(detections)[:, :4])
    d_classes = np.array(detections)[:, -1].astype(int)
    print('IMAGE:',image_name, 11 in d_classes)
    if image_name == '20220706235557382UTC_capture.jpg':
        pdb.set_trace()
    detections = detections[np.where(d_confs > conf_thresh)]
    labels_detected = np.zeros(len(labels))
    detections_matched = np.zeros(len(detections))
    for l_idx, (l_class, l_bbox) in enumerate(zip(l_classes, l_bboxs)):
        for d_idx, (d_bbox, d_class) in enumerate(zip(d_bboxs, d_classes)):
            iou = pairwise_iou(Boxes(torch.from_numpy(np.array([l_bbox]))), Boxes(torch.from_numpy(np.array([d_bbox]))))
            if 11 in d_classes:
                print('PAIRWISE IOUS:')
            if iou >= iou_thresh:
                if 11 in d_classes:
                    print('GREATER THAN IOU THRESH')
                    pdb.set_trace()
                confusion_matrix[l_class, d_class] += 1
                labels_detected[l_idx] = 1
                detections_matched[d_idx] = 1
    for i in np.where(labels_detected == 0)[0]:
        confusion_matrix[l_classes[i], -1] += 1
    for i in np.where(detections_matched == 0)[0]:
        confusion_matrix[-1, d_classes[i]] += 1
    return confusion_matrix

def main(args):
    """Trains, evals, or predicts using the Detectron2 model."""

    # Merges configs with Detectron2 defaults.
    cfg = load_cfg(args)
    default_setup(cfg, args)

    if args.train:  
        for x in range(len(args.test_dir)):
            _, _ = register_dataset(
            cfg.DATASETS.TEST[x],
            args.test_json[x],
            args.test_dir[x],
        ) 
        for x in range(len(args.train_dirs)):
            _, _ = register_dataset(
            cfg.DATASETS.TRAIN[x],
            args.train_jsons[x],
            args.train_dirs[x],
        ) 
        with open(osp.join(args.output_dir,'training_config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        print("IN THE TRAININGGGGGGGGGGGGGGG")
        if args.transformer:
            trainer = COCOAzureTransformerTrainer(cfg)
            trainer.resume_or_load(resume=args.resume)
            trainer.train()
        else:
            trainer = COCOAzureTrainer(cfg)
            trainer.resume_or_load(resume=args.resume)
            trainer.train()
    
    else:
        if args.transformer:
            if args.eval:        
                print("IN THE EVALUATIONNNNNNNNNNNNNN")
                predictor = DefaultPredictor(cfg)
                test_dict, test_metadata = register_dataset( cfg.DATASETS.TEST[0],  args.test_json[0],  args.test_dir[0],)
                evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir=args.output_dir )
                test_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])#, sampler=InferenceSampler(1))
                metrics_dict = inference_on_dataset(predictor.model, test_loader, evaluator)
                with open(osp.join(args.output_dir,'eval_metrics_{0}.json'.format(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)), 'w') as f:
                    json.dump(metrics_dict, f)
                output_writing = "{0} {1} {2}".format(metrics_dict["bbox"]["AP"], metrics_dict["bbox"]["AP50"], metrics_dict["bbox"]["AP75"])
                with open(osp.join(args.output_dir,'eval_metrics_{0}.txt'.format(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)), 'w') as f:
                    json.dump(output_writing, f)
                print("OUTPUT OF METRICS DICTTTTTTTTTTTTTTT")
                print(metrics_dict)
                writer = Run.get_context()
                writer.log("Test/AP", np.float(metrics_dict["bbox"]["AP"]))
                writer.log("Test/AP50", np.float(metrics_dict["bbox"]["AP50"]))
                writer.log("Test/AP75", np.float(metrics_dict["bbox"]["AP75"]))
        else:
            if args.eval:        
                print("IN THE EVALUATIONNNNNNNNNNNNNN")
                predictor = DefaultPredictor(cfg)
                test_dict, test_metadata = register_dataset( cfg.DATASETS.TEST[0],  args.test_json[0],  args.test_dir[0],)
                evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir=args.output_dir )
                test_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])#, sampler=InferenceSampler(1))
                metrics_dict = inference_on_dataset(predictor.model, test_loader, evaluator)
                with open(osp.join(args.output_dir,'eval_metrics_{0}.json'.format(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)), 'w') as f:
                    json.dump(metrics_dict, f)
                output_writing = "{0} {1} {2}".format(metrics_dict["bbox"]["AP"], metrics_dict["bbox"]["AP50"], metrics_dict["bbox"]["AP75"])
                with open(osp.join(args.output_dir,'eval_metrics_{0}.txt'.format(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)), 'w') as f:
                    json.dump(output_writing, f)
                print("OUTPUT OF METRICS DICTTTTTTTTTTTTTTT")
                print(metrics_dict)
                writer = Run.get_context()
                writer.log("Test/AP", np.float(metrics_dict["bbox"]["AP"]))
                writer.log("Test/AP50", np.float(metrics_dict["bbox"]["AP50"]))
                writer.log("Test/AP75", np.float(metrics_dict["bbox"]["AP75"]))

            if not args.unlabled_predict:
                try:
                    test_dict, test_metadata = register_dataset(
                        cfg.DATASETS.TEST[0],
                        args.test_json[0],
                        args.test_dir[0],
                    ) 
                except:
                    print("already registered?")  
            if args.predict:
                print("IN THE PREDICTIONNNNNNNNNNNNNNN")
                os.makedirs(osp.join(osp.join(args.output_dir,'visualizations'),'predictions_70'), exist_ok=True)
                test_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])#, sampler=InferenceSampler(args.grad_cam))
                predictor = DefaultPredictor(cfg)
                for num,d in zip(range(len(test_dict)),tqdm(test_loader)):
                    name = d[0]["file_name"].split("/")[-1]
                    img = cv2.imread(d[0]["file_name"])
                    outputs = predictor(img)
                    v = Visualizer(img[:,:,::-1], metadata=test_metadata)
                    v.draw_dataset_dict(test_dict[num])
                    v2 = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'predictions_70'), f"{name}"),v2.get_image()[:,:,::-1],)
            if args.unlabled_predict:
                print("IN THE UNLABLED PREDICITON")
                predictor = DefaultPredictor(cfg)
                for filename in tqdm(glob.glob(f"{args.test_dir[0]}/*.jpg")):
                    name = filename.split("/")[-1]
                    img = cv2.imread(filename)
                    outputs = predictor(img)
                    os.makedirs(osp.join(osp.join(args.output_dir,'visualizations'),'unlabeled_predictions'), exist_ok=True)
                    v = Visualizer(img[:,:,::-1])
                    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'unlabeled_predictions'), f"{name}"),v.get_image()[:,:,::-1],)

            if args.visualize:
                print("IN THE VISUALIZATIONNNNNNNNNNNNN")
                os.makedirs(osp.join(args.output_dir,'visualizations'), exist_ok=True)
                predictor = DefaultPredictor(cfg)
                if args.tsne:
                    feature_vectors = None
                    labels = []
                    layer_name = "roi_heads.box_head" 
                    os.makedirs(osp.join(osp.join(args.output_dir,'visualizations'),'tsne/{0}'.format(layer_name)), exist_ok=True)
                    grad_cam = GradCAM(predictor.model, layer_name)
                    for d in tqdm(test_dict):
                        # Loads image using cv2.
                        original_image = cv2.imread(d["file_name"])
                        raw_height, raw_width = original_image.shape[:2]
                        image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))  # ndarray to tensor and (h,w,c) to (c,h,w)
                        inputs = {"image": image, "height": raw_height, "width": raw_width}
                        feature_vec, class_id = grad_cam(inputs, type='tsne')  # cam mask
                        feature_vectors = feature_vec if feature_vectors is None else torch.vstack((feature_vectors, feature_vec))
                        labels.append(class_id)
                    grad_cam._release_activations_grads()
                    utils.tsne_viz_feature_vecs(feature_vectors, labels, osp.join(osp.join(args.output_dir,'visualizations/tsne/{0}'.format(layer_name)), 'roi_pooling_tsne_features'))
                if args.grad_cam != -1:
                    layer_name = args.layer_name #"backbone.bottom_up.res5.2.conv3"
                    os.makedirs(osp.join(osp.join(args.output_dir,'visualizations'),'gradcam/{0}'.format(layer_name)), exist_ok=True)
                    grad_cam = GradCAM(predictor.model, layer_name)
                    if len(args.specific_image) > 0:
                        for name in args.specific_image:
                            # pdb.set_trace()
                            for d in test_dict:
                                if d["file_name"].split("/")[-1] == name:
                                    break
                            original_image = cv2.imread(d["file_name"])
                            raw_height, raw_width = original_image.shape[:2]
                            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))  # ndarray to tensor and (h,w,c) to (c,h,w)
                            inputs = {"image": image, "height": raw_height, "width": raw_width}
                            # Grad-CAM
                            cam, cam_orig, output, target_cateogry = grad_cam(inputs, target_category=args.grad_cam, type='gradcam')
                            v = Visualizer(original_image[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.0)
                            out = v.draw_instance_predictions(output[0]["instances"].to("cpu"))
                            cam_arr8 = (cam*256).astype(np.uint8)
                            out_arr8 = (out.get_image()).astype(np.uint8)[:,:,::-1]
                            heatmap = cv2.applyColorMap(cam_arr8, cv2.COLORMAP_JET)
                            plot_image = cv2.addWeighted(out_arr8, 0.5, heatmap, 1 - 0.5, 0)
                            cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'gradcam'), '{0}/grad_cam_cat_{1}_{2}'.format(layer_name, target_cateogry, name)), plot_image)
                    else:
                        for d in tqdm(test_dict):
                            name = d["file_name"].split("/")[-1]
                            original_image = cv2.imread(d["file_name"])
                            raw_height, raw_width = original_image.shape[:2]
                            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))  # ndarray to tensor and (h,w,c) to (c,h,w)
                            inputs = {"image": image, "height": raw_height, "width": raw_width}
                            # Grad-CAM
                            cam, cam_orig, output, target_cateogry = grad_cam(inputs, target_category=args.grad_cam, type='gradcam')
                            v = Visualizer(original_image[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.0)
                            out = v.draw_instance_predictions(output[0]["instances"].to("cpu"))
                            cam_arr8 = (cam*256).astype(np.uint8)
                            out_arr8 = (out.get_image()).astype(np.uint8)[:,:,::-1]
                            heatmap = cv2.applyColorMap(cam_arr8, cv2.COLORMAP_JET)
                            plot_image = cv2.addWeighted(out_arr8, 0.5, heatmap, 1 - 0.5, 0)
                            cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'gradcam'), '{0}/grad_cam_cat_{1}_{2}'.format(layer_name, target_cateogry, name)), plot_image)
                    grad_cam._release_activations_grads()
                if args.synth_grad_cam != -1:
                    layer_name = args.layer_name #"backbone.bottom_up.res5.2.conv3"
                    os.makedirs(osp.join(osp.join(args.output_dir,'visualizations'),'gradcam/{0}'.format(layer_name)), exist_ok=True)
                    grad_cam = GradCAM(predictor.model, layer_name)
                    onlyfiles = [f for f in os.listdir(test_metadata.image_root) if osp.isfile(osp.join(test_metadata.image_root, f))]
                    for name in tqdm(onlyfiles):
                        found = False
                        if name [-4:] == 'json':
                            continue
                        for d in test_dict:
                            if d["file_name"].split("/")[-1] == name:
                                found = True
                                break
                        if not found:
                            print('COULD NOT FIND THIS FILE', name)
                            continue
                        original_image = cv2.imread(d["file_name"])
                        raw_height, raw_width = original_image.shape[:2]
                        image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))  # ndarray to tensor and (h,w,c) to (c,h,w)
                        inputs = {"image": image, "height": raw_height, "width": raw_width}
                        # Grad-CAM
                        cam, cam_orig, output, target_cateogry = grad_cam(inputs, target_category=args.synth_grad_cam, type='gradcam')
                        v = Visualizer(original_image[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.0)
                        out = v.draw_instance_predictions(output[0]["instances"].to("cpu"))
                        cam_arr8 = (cam*256).astype(np.uint8)
                        out_arr8 = (out.get_image()).astype(np.uint8)[:,:,::-1]
                        heatmap = cv2.applyColorMap(cam_arr8, cv2.COLORMAP_JET)
                        plot_image = cv2.addWeighted(out_arr8, 0.5, heatmap, 1 - 0.5, 0)
                        cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'gradcam'), '{0}/grad_cam_cat_{1}_{2}'.format(layer_name, target_cateogry, name)), plot_image)
                if args.guided_gradcam != -1:
                    layer_name = args.layer_name #"backbone.bottom_up.res5.2.conv3"
                    os.makedirs(osp.join(osp.join(args.output_dir,'visualizations'),'guided_gradcam/{0}'.format(layer_name)), exist_ok=True)
                    grad_cam = GradCAM(predictor.model, layer_name)
                    GBP = GuidedBackprop(predictor.model, layer_name)
                    if len(args.specific_image) > 0:
                        for name in args.specific_image:
                            # pdb.set_trace()
                            for d in test_dict:
                                if d["file_name"].split("/")[-1] == name:
                                    break
                            original_image = cv2.imread(d["file_name"])
                            raw_height, raw_width = original_image.shape[:2]
                            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))  # ndarray to tensor and (h,w,c) to (c,h,w)
                            inputs = {"image": image, "height": raw_height, "width": raw_width}
                            target_cateogry = args.guided_gradcam
                            # Grad cam
                            cam, cam_orig, output, target_cateogry = grad_cam(inputs, target_category=target_cateogry, type='gradcam')
                            cam = (cam*256).astype(np.uint8)
                            print('Grad cam completed')
                            # Guided backprop
                            guided_grads, target_cateogry = GBP.generate_gradients(inputs, target_cateogry)
                            guided_grads = (guided_grads*255).astype(np.uint8)
                            print('Guided backpropagation completed')
                            # Guided Grad cam
                            cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'guided_gradcam'), '{0}/cam_{1}'.format(layer_name, name, target_cateogry)), cam)
                            cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'guided_gradcam'), '{0}/guided_grads_{1}'.format(layer_name, name, target_cateogry)), guided_grads)
                            cam_gb = np.multiply(cam, guided_grads)
                            cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'guided_gradcam'), '{0}/guided_grad_cam_{1}'.format(layer_name, name, target_cateogry)), cam_gb)
                    else:
                        for d in tqdm(test_dict):
                            name = d["file_name"].split("/")[-1]
                            original_image = cv2.imread(d["file_name"])
                            raw_height, raw_width = original_image.shape[:2]
                            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))  # ndarray to tensor and (h,w,c) to (c,h,w)
                            inputs = {"image": image, "height": raw_height, "width": raw_width}
                            target_cateogry = args.guided_gradcam
                            # Grad cam
                            cam, cam_orig, output, target_cateogry = grad_cam(inputs, target_category=target_cateogry, type='gradcam')
                            print('Grad cam completed')
                            # Guided backprop
                            guided_grads, target_cateogry = GBP.generate_gradients(inputs, target_cateogry)
                            print('Guided backpropagation completed')
                            # Guided Grad cam
                            pdb.set_trace()
                            cam_gb = (np.multiply(cam_orig, guided_grads)*256).astype(np.uint8)
                            cam_gb = cv2.resize(cam_gb, (inputs['width'], inputs['height']))
                            cam = (cam*256).astype(np.uint8)
                            cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'guided_gradcam'), '{0}/cam_{1}'.format(layer_name, name, target_cateogry)), cam)
                            cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'guided_gradcam'), '{0}/guided_grads_{1}'.format(layer_name, name, target_cateogry)), guided_grads)
                            cv2.imwrite(osp.join(osp.join(osp.join(args.output_dir,'visualizations'),'guided_gradcam'), '{0}/guided_grad_cam_{1}'.format(layer_name, name, target_cateogry)), cam_gb)
                    grad_cam._release_activations_grads()
            if args.cm:
                print("CONFUSION MATRIXXXXXXXX")
                os.makedirs(osp.join(args.output_dir,'confusion_matrix'), exist_ok=True)
                n_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
                confusion_matrix = np.zeros([n_classes + 1, n_classes + 1])
                predictor = DefaultPredictor(cfg)
                for d in tqdm(test_dict):
                    name = d["file_name"].split("/")[-1]
                    original_image = cv2.imread(d["file_name"])
                    outputs = predictor(original_image)
                    labels = list()
                    detections = list()
                    for coord, conf, cls, ann in zip(
                        outputs["instances"].get("pred_boxes").tensor.cpu().numpy(),
                        outputs["instances"].get("scores").cpu().numpy(),
                        outputs["instances"].get("pred_classes").cpu().numpy(),
                        d["annotations"]
                    ):
                        labels.append([ann["category_id"]] + ann["bbox"])
                        detections.append(list(coord) + [conf] + [cls])
                    try:
                        confusion_matrix += conf_matrix_calc(name, np.array(labels), np.array(detections), n_classes, conf_thresh=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, iou_thresh=0.5)
                    except:
                        pdb.set_trace()
                        
                matrix_indexes = test_metadata.get("thing_classes") + ["null"]
                pdb.set_trace()
                df = pd.DataFrame(confusion_matrix, columns=matrix_indexes, index=matrix_indexes)
                filepath = osp.join(args.output_dir,'confusion_matrix/confusion_matrix.csv')
                df.to_csv(filepath)

if __name__ == "__main__":
    # Parses input arguments.
    args = parse_args()
    if args.transformer:
        print('OUTPUT DIR',args.output_dir)
        args.output_dir = 'transformer_' + args.output_dir
    # Sets phillytools dirs.
    args = set_pt_dirs(args)
    
    # Makes output directory.
    os.makedirs(args.output_dir, exist_ok=True)
    # Launches distributed Detectron2 task on GPUs.
    launch(main, args.gpus, dist_url="auto", args=(args,))

