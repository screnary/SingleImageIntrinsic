# BIMEF

![](https://img.shields.io/badge/MATLAB-R2016b-green.svg) 

![](https://img.shields.io/badge/OS-Win10-green.svg) 

Code for our paper "A Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement"

* The code for the comparison method is also provided, see [lowlight](https://github.com/baidut/BIMEF/tree/master/lowlight)
* Downloads: [google Drive](https://drive.google.com/drive/folders/0B_FjaR958nw_djVQanJqeEhUM1k?usp=sharing)  (Just unzip data to current folder)
  * Datasets `VV, LIME, NPE, NPE-ex1, NPE-ex2, NPE-ex3, DICM, MEF`
  * Since some methods are quite time-consuming, we also provide their results (e.g. `results__dong@VV.zip`)
  * Since some metrics are quite time-consuming, we also provide their results (`TestReport.zip`)
* All the experiments can be reproduced easily by running `experiments.m`

![tcyb2017_moreExamples](example.jpg)

From left to right: input images, results of MSRCR, Dong, NPE, LIME, MF, SRIE, and BIMEF(ours).

## Datasets

- [VV](https://sites.google.com/site/vonikakis/datasets) （**Busting image enhancement and tone-mapping algorithms: **A collection of the most challenging cases）
- [LIME-data](http://cs.tju.edu.cn/orgs/vision/~xguo/LIME.htm)
- [NPE-data, NPE-ex1, NPE-ex2, NPE-ex3](http://blog.sina.com.cn/s/blog_a0a06f190101cvon.html)
- DICM —— 69 captured images from commercial digital cameras: [Download (15.3 MB)](http://mcl.korea.ac.kr/projects/LDR/LDR_TEST_IMAGES_DICM.zip)
- [MEF](https://ece.uwaterloo.ca/~k29ma/)  [dataset](http://ivc.uwaterloo.ca/database/MEF/MEF-Database.php)

## Prerequisites

* Original code is tested on *Matlab 2016b* 64bit, Windows 10. 
* [matlabPyrTools](https://github.com/gregfreeman/matlabPyrTools) is required to run VIF metric (`vif.m`).

## Setup

Run `startup.m` to add required path, then you are able to try the following demo.

```matlab
I = imread('yellowlily.jpg');
J = BIMEF(I); 
subplot 121; imshow(I); title('Original Image');
subplot 122; imshow(J); title('Enhanced Result');
```

Replace `BIMEF` with other methods you want to test.

## Directory Structure

```
.
├── data           # put your datasets here
│   ├── MEF        # dataset name (VV, LIME, NPE...)
│        ├── out   
│        │    ├── loe100x100           # LOE visualization results
│        │    ├── TestReport.csv       # results of metrics
│        │    ├── TestReport__xxxx.csv # backups of TestReport
│        │    └── xxx__method.PNG      # output images
│        └── xxx.jpg                   # input images
│
├── lowlight       # lowlight image enhancement methods
├── quality        # image quality metrics (blind or full-reference)
├── util           # provide commonly used utility functions
│
├── demo.m         # simple demo of lowlight enhancement
├── experiments.m  # reproduce our experiments
└── startup.m      # for installation
```

## Usage

Run experiments.

```matlab
% specify datasets
dataset = {'VV' 'LIME' 'NPE' 'NPE-ex1' 'NPE-ex2' 'NPE-ex3' 'MEF' 'DICM'};
dataset = strcat('data', filesep, dataset, filesep, '*.*');

% specify methods and metrics
method = {@multiscaleRetinex @dong @npe @lime @mf @srie @BIMEF};
metric = {@loe100x100 @vif};

for d = dataset, data = d{1};
    data,  
    Test = TestImage(data);        
    Test.Method = method; 
    Test.Metric = metric;
    
    % run test and display results
    Test,                     
    
    % save test to a .csv file
    save(Test);
end
```

Show test reports.

```matlab
% table
TestReport('TestReport__VV.csv'),

% boxplot
TestReport('TestReport__MEF.csv').boxplot;
```

Our method (BIMEF) has the lowest LOE and the highest VIF.

![boxplot](boxplot.jpg)