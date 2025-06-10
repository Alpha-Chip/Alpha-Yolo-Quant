# 🤖 YOLOv8n quantisation 🤖
The YOLOv8n quantisation project is an implementation of a computer vision algorithm for object detection that
uses quantisation to reduce the model size without significant loss of accuracy. The main goal of the project is to.
create quantised weights for Verilog implementations of a compressed model architecture capable of running on devices with limited computational resources.


## Table of Contents
- [Input and output data](#Input-and-output-data)
- [Technologies](#Technologies)
- [Utilisation](#Utilisation)
- [Description and order of executable files](#Description-and-order-of-executable-files)

## Input and output data
### Input
- model yolov8n.pt
- bit number 
### Output
- original weights .pickle
- weights batchnormfusion .pickle
- compressed architecture yolov8n
- quantised activations
- Verilog format quantised weights

## Technologies
- Python
- Numpy
- PyTorch
- Pandas
- Matplotlib
- PIL

## Utilisation
#### Repository cloning:
```sh
git clone https://github.com/sopheroner/yolov8n_quantisation.git
```
#### Dependency installation
```sh
pip install -r requirements.txt
```

## Description and order of executable files
0. ```stage_0.py``` - config with initial parameters
1. ```stage_1.py``` - model preparation. From the ready model (yolov8n.pt) it is necessary to get weights with names that match the names of weights in the custom architecture
2. ```stage_2.py``` - applying the BatchNormFusion method to the architecture (all BatchNorm layers become one with conv)
3. ```stage_3.py``` - validation of the initial model (COCO dataset)
4. ```stage_4.py``` - model validation with BatchNormFusion weights + calibration. It is necessary to compare the metrics of the original model and the batchnormf model (should be the same / slightly different). Calibration - maximum value of activations
5. ```stage_5.py``` - maximum selection
6. ```stage_6.py``` - quantisation of weights. Conversion of weights dimension from float to int (without last layers)
7. ```stage_6_full_quant.py``` - quantisation of weights (including last layers)
8. ```stage_7.py``` - formatting quantised weights for yolo architecture
9. ```stage_8_torch.py``` - validation of the quantised model WITHOUT q_NMS
9. ```stage_8_torch_full_quant.py``` - Validation of the quantised C q_NMS model
