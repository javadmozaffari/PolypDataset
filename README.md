# A novel dataset for polyp segmentation and detection using a vision-language model
### This repository contains the code and dataset for the paper "A novel dataset for polyp segmentation and detection using a vision-language model".

## Proposed Network

The proposed model features a dual-encoder architecture designed to extract complementary features from colonoscopy images and their associated clinical reports. A Vision Transformer (ViT) structure was used to effectively integrate the features from both encoders. The decoder includes two innovative components: the Global Attention Module (GAM), which captures global contextual information, and the Multi-Scale Aggregation Module (MSAM), which aggregates features across different scales. These modules work together to enhance the model's ability to capture both local details and global context, enabling precise and comprehensive segmentation of polyps in colonoscopy images.<br><br>
<div align="center">
  <img src="https://github.com/javadmozaffari/PolypDataset/blob/main/Image/Overview.jpg" alt="alt text" height="400"/>
</div><br>

## Dataset 
The dataset consists of 740 colonoscopy images, each with a resolution of 480×720, collected from 293 patients at a medical center. Binary masks and bounding boxes were generated for these images using annotation tools such as Label-studio and Yolo-label. To ensure high quality, all images underwent manual review, and low-quality samples were excluded. Women account for 55.2% of the patients, with an average age of 56.7 years, while men have an average age of 53.2 years. The age of patients ranges from 14 to 83 years, with the 50–60 age group being the most represented, comprising 28% of the total. Furthermore, the dataset provides statistical analyses of polyp size and location, enhancing its utility for clinical and research purposes. The dataset is available on [Google Drive](https://drive.google.com/file/d/1iwCvbIxOTC1kE8wUX6xct5D5YF--li0u/view?usp=sharing). To facilitate a quick start, we have included a sample demonstration from the polyp dataset.

PolypDataset:  
    
    |-- Train_Folder
    |   |-- imgs
    |   |   |-- xxx.jpg
    |   |   |-- ....
    |   |   |-- xxx.jpg
    |   |-- labels
    |   |   |-- xxx.png
    |   |   |-- ....
    |   |   |-- xxx.png
    |   |-- Reports.xlsx
    |-- Test_Folder
    |   |-- imgs
    |   |   |-- xxx.jpg
    |   |   |-- ....
    |   |   |-- xxx.jpg
    |   |-- labels
    |   |   |-- xxx.png
    |   |   |-- ....
    |   |   |-- xxx.png
    |   |-- Reports.xlsx

## Comprehensive Benchmark
To comprehensively evaluate and advance polyp analysis, we provide a benchmark encompassing both segmentation and detection tasks.
### Segmentation
Our model was compared with nine previous models, including UNet, PSPNet, DeepLabV3+, MetaUnet, ColonGen, PolypPVT, Colonformer, TGANet, and LViT. The first version of our model achieved a 2.7% and 2.5% improvement in Dice and IoU scores, respectively, as presented in Table 2. When textual information was incorporated alongside images, Dice and IoU further improved by 0.7% and 0.5%, respectively. To comprehensively evaluate segmentation performance, we selected seven diverse samples from the dataset, encompassing polyps of various numbers, sizes, and shapes.<br><br> 
<div align="center">
  <img src="https://github.com/javadmozaffari/PolypDataset/blob/main/Image/Segmentation.jpg" alt="alt text" height="400"/>
</div><br>

### Detection
For the detection task, we evaluated several recent state-of-the-art object detection models. The results indicate that YOLOv9e and YOLO-World deliver the best performance in detecting small polyps. Conversely, YOLOv8x demonstrated comparatively lower effectiveness in identifying small polyps. This evaluation included various types of polyps to thoroughly assess each model's performance.<br><br>
<div align="center">
  <img src="https://github.com/javadmozaffari/PolypDataset/blob/main/Image/Detection.jpg" alt="alt text" height="400"/>
</div><br>
