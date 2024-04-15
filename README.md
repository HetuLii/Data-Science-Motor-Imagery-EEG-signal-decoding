# Introduction
This repository contains a replication study of several baseline and state-of-the-art models focusing on motor imagery EEG signal decoding. Further, experiments on some unaddressed questions are carried out and recorded. This project aims to:
* Provide pre-processed, augmented, and ready-to-use motor imagery EEG dataset **BCIC-IV-2A**
* Compare the cross-subject performances of models including fundamental **CNN**, **LSTM** models, hybrid models, **EEGNet-8,2** model [4], and **EEGNeX** model [1] in pytorch
* Enable other researchers to conveniently carry out cross-subject comparisons on their datasets and models
* Investigate the cross-subject efficacy of incorporating **Attention layer** in the EEGNet-8,2 model 

# Dataset: BCIB-IV-2A
## Dataset Description
**The dataset can be downloaded at: https://drive.google.com/drive/folders/1NJPo63BcCKZNOd-rcsdNqzCEgS5X_Jcm?usp=sharing**
The **BCIC-IV-2A** dataset is recorded from $9$ subjects. The cue-based BCI paradigm consisted the imagination of movement of the left hand (class 1), right hand (class 2), both feet (class 3), and tongue (class 4). Two sessions on different days were recorded for each subject. Each session is comprised of $6$ runs separated by short breaks. 
One run consists of $48$ trials ($12$ for each of the four possible classes), yielding a total of $288$ trials per session.

For instance, the data for the first participant are provided in two files labeled as A01T/A01E for the two complete sessions. Each file contains $288$ samples, where 'T' is used for training and 'E' for testing.

The data are collected at a rate of $250 Hz$. The CSV file represents $4$ seconds of data, specifically the data from $2$ to $6$ seconds in **Figure 1** [2], totaling $1000$ data points. The data underwent bandpass filtering from $0.5$ to $100 Hz$ and a notch filter at $50 Hz$ to eliminate noise interference.

The dataset utilizes the **international 10-20 system** for electrode placement. There are $25$ channels in total, of which three channels are for Electrooculography (EOG) data. For classifying MI (Motor Imagery) signals, it is necessary to filter the channels during preprocessing, retaining $22$ channels. The electrode montage is shown in **Figure 2**.[2]

![alt text](https://github.com/HetuLii/Data-Science-Motor-Imagery-EEG-signal-decoding/blob/main/images/Cue-based%20BCI%20paradigm.png)
*Figure 1: Timing scheme of the cue-based BCI paradigm*
![alt text](https://github.com/HetuLii/Data-Science-Motor-Imagery-EEG-signal-decoding/blob/main/images/Electrode%20montage.png)
*Figure 2: Left: Electrode montage corresponding to the international 10-20 system. Right: Electrode montage of the three monopolar EOG channels*
# data_preprocessing.py
## Pre-Processing
Before any processing, each subject gives a dataset with shape $(576, 22000)$. We can impose a butterworth filter ($0.1 Hz$ to $39 Hz$) to reduce noise and irrelevant information. To avoid vanishing/exploding gradient problems, we impose Min-Max Scaling. We then need to expand the dimension such that various models can be trained on the dataset. 
We first reshape the model to the shape $(576, 22, 1000)$, where axis $1$ denotes the number of electrodes (channels), and the last axis denotes the number of samplings per electrode ($4 \cdot 250 = 1000$). This is already sufficient for models such as Long Short-Term Memory (LSTM) and 1D-CNN model. For models involving 2D-CNN, we need to upscale the dataset again to shape $(576, 1, 22, 1000)$.
## Corss-subject dataset
To check the cross-subject performances of the models, we need to choose one subject and take its dataset as the testing dataset, which will include $576$ data points. The rest eight subjects contribute to the training dataset, which thus includes $4608$ data points. Then, we split them into batches with size $32$, which is for parallelism during optimization. 
# Models.py
The file *Models.py* includes models: **LSTM**, **1D-CNN**, **2D-CNN**, **CNN-LSTM**, **CNN-GRU**, **EEGNet-8-2**, **EEGNet-Attention**, and **EEGNeX** in sequence. The LSTM and 1D-CNN model extract only temporal features. Hybrid models including CNN-LSTM and CNN-GRU extract first the local spatial features using CNN layers and the temporal features using LSTM/GRU layer. SOTA models such as EEGNet-8-2 model and EEGNeX model are more complex and extract both temporal and spatial features using CNN layers. The model architectures of EEGNet-8-2 model and EEGNeX model are shown below in **Figure 3-4**.
![alt text](https://github.com/HetuLii/Data-Science-Motor-Imagery-EEG-signal-decoding/blob/main/images/EEGNet.png)
*Figure 3: EEGNet model architecture*
![alt text](https://github.com/HetuLii/Data-Science-Motor-Imagery-EEG-signal-decoding/blob/main/images/EEGNeX.png)
*Figure 4: EEGNeX model architecture*
# Results 
**The list of $72$ confusion matrices are included in:** *list_of_results.pkl* in the images folder.

Training and testing all the $8$ models, we reach the results shown in **Table 1**. We can clearly observe that there are big differences both in-between the subects and in-between the models, and the **EEGNeX** model achieves the best accuracies for tests on seven out of nine subjects. We then make a few model comparisons and further analysis. 
<div align="center">

| | LSTM | 1D-CNN | 2D-CNN | CNN-LSTM | CNN-GRU | EEGNet-8-2 | EEGNet-Attention | EEGNeX |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| s1 | 0.396 | 0.556 | 0.568 | 0.608 | 0.609 | 0.601 | 0.387 | **0.672** |
| s2 | **0.306** | 0.262 | 0.285 | 0.252 | 0.276 | 0.253 | 0.238 | 0.240 |
| s3 | 0.401 | 0.536 | 0.509 | 0.549 | 0.571 | 0.738 | 0.439 | **0.786** |
| s4 | 0.278 | 0.380 | 0.394 | 0.384 | 0.406 | 0.464 | 0.314 | **0.469** |
| s5 | 0.252 | 0.307 | 0.330 | 0.288 | 0.321 | 0.252 | 0.274 | **0.354** |
| s6 | 0.288 | 0.326 | 0.330 | 0.313 | 0.323 | 0.316 | 0.281 | **0.380** |
| s7 | 0.345 | 0.417 | 0.389 | 0.444 | 0.436 | **0.497** | 0.288 | 0.495 |
| s8 | 0.314 | 0.563 | 0.604 | 0.580 | 0.599 | 0.618 | 0.365 | **0.722** |
| s9 | 0.337 | 0.479 | 0.500 | 0.512 | 0.535 | 0.566 | 0.349 | **0.660** |

*Table 1: Summary of the accuracies of different models with different validation subjects*

</div>

## 1D-CNN model "versus" 2D-CNN model
From previous experiments, it has been observed that 2D-CNN models are more effective than 1D-CNN models in within-subject classifications [1]. In our cross-subject experiment, the mean accuracy of the 1D-CNN model was $0.425$, while that of the 2D-CNN model was $0.434$. There is not sufficient statistical evidence to conclude that the 2D-CNN model is better than the 1D-CNN model, as the p-value is $0.17$, which exceeds the common significance threshold of $0.05$. A possible reason for this could be that cross-channel interactions, which are effectively modeled by the 2D-CNN, may vary significantly across subjects, thus diminishing the advantages of 2D-CNN model.

## 2D-CNN model "versus" Hybrid model
From previous experiments, it has been observed that hybrid models (CNN-LSTM, CNN-GRU) are more effective than 2D-CNN models in within-subject classifications [1]. In our experiment, both CNN-LSTM ($0.437$) and CNN-GRU ($0.453$) achieved higher mean accuracies than the 2D-CNN ($0.434$). However, there is sufficient statistical evidence only for the CNN-GRU model being superior to the 2D-CNN model under a significance threshold of $0.05$ (p-value $0.0276$). The key takeaways are that it remains beneficial to extract additional temporal information after appropriate spatial filtering or learning in a cross-subject scenario. **The GRU model performs better**, likely due to it having fewer parameters and thus reducing the risk of overfitting.

## EEGNet with Attention
From previous experiments, it was stated that Attention mechanism can make up the limitations of CNN in perceiting global dependencies, and applying the locally extracted features to the Attention mechanism can improve the within-subject classifications [3]. However, in our cross-subject experiment, EEGNet-Attention model reaches mean accuracy of $0.326$, while EEGNet alone reaches accuracy of $0.478$. There is sufficient evidence for the EEGNet model being better than EEGNet-Attention model. The p-value is negligible. The results show that the global dependencies learnt by the Attention mechanism **cannot** be applied well to different subjects.

## Difference between subjects
We can observe that there are big differences in accuracies for tests on different subjects. For instance, the accuracies are particularly low for subject 2,5,6. It's interesting to observe that the simpler model **LSTM** has the best accuracy for subject 2. This implicate that the cross-channel interaction modelling in, for instance, EEGNet and EEGNeX models, fails on subject 2. It is very insteresting but out of the project's working scope to study further the brain state of subject 2 and find the causes of the failings. But this result can at least inform us that the superioty of CNN-based model over LSTM model [1] may not be always true in cross-subject scenarios.

## Confusion Maps
It is insightful to examine the confusion matrices of various models and subjects. We present four such matrices here in **Figure 5-8**. Notably, contrary to the expected confusion between left-hand and right-hand imagery, misclassifications frequently involve identifying hand movements as feet or tongue imagery motions, which is counterintuitive. This trend is consistent across the subjects and models. 
![alt text](https://github.com/HetuLii/Data-Science-Motor-Imagery-EEG-signal-decoding/blob/main/images/Confusion_S2.png)
*Figure 5: Confusion map for the subject 2*
![alt text](https://github.com/HetuLii/Data-Science-Motor-Imagery-EEG-signal-decoding/blob/main/images/Confusion_S3.png)
*Figure 6: Confusion map for the subject 3*

Additionally, it is worth mentioning that the EEGNeX model, compared to the LSTM model, makes fewer errors in classifying left-hand motions as feet, and feet as tongue, and vice versa. This observation suggests that **imagery of feet and tongue motions may elicit learnable cross-channel patterns**, offering an avenue for further research.
![alt text](https://github.com/HetuLii/Data-Science-Motor-Imagery-EEG-signal-decoding/blob/main/images/Confusion_LSTM.png)
*Figure 7: Confusion map for the LSTM model*
![alt text](https://github.com/HetuLii/Data-Science-Motor-Imagery-EEG-signal-decoding/blob/main/images/Confusion_NeX.png)
*Figure 8: Confusion map for the EEGNeX model*

# Conclusions
Cross-subject experiments on the motor imagery EEG dataset BCIC-IV-2A gives some different conclusions than within-subject ones provided by previous studies. 
* There is not enough evidence to show that 2D-CNN model is better than 1D-CNN model under $0.05$ significance level. 
* CNN-GRU model is better than the 2D-CNN model and CNN-LSTM model under cross-subject scenario.
* Adding Attention layer after CNN layers does not work well under cross-subject scenario.
* Cross-subject misclassifications frequently involve identifying hand movements as feet or tongue imagery motions.
* Imagery of feet and tongue motions may elicit learnable cross-channel patterns that are consistent over subjects.
  
# References
[1]: **Chen et al.(2023)**, Toward Reliable Signals Decoding for Electroencephalogram: A Benchmark Study to EEGNeX

[2]: **Brunner et al.(2008)**, BCI Competition 2008 – Graz data set A

[3]: **Liu et al.(2023)**, A study of EEG classification based on attention mechanism and EEGNet Motor Imagination

[4]: **Lawhern et al.(2018)**, EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces
