# Introduction
This repository contains a replication study of several baseline and state-of-the-art models focusing on motor imagery EEG signal decoding. Further, experiments on some unaddressed questions are carried out and recorded. This project aims to:
* Provide pre-processed, augmented, and ready-to-use motor imagery EEG dataset **BCIC-IV-2A**
* Compare the cross-subject performances of models including fundamental **CNN**, **LSTM** models, hybrid models, **EEGNet-8,2** model [4], and **EEGNeX** model [1] in pytorch
* Enable other researchers to conveniently carry out cross-subject comparisons on their datasets and models
* Investigate the cross-subject efficacy of incorporating **Attention layer** in the EEGNet-8,2 model 

# Dataset: BCIB-IV-2A
## Dataset Description
The **BCIC-IV-2A** dataset is recorded from $9$ subjects. The cue-based BCI paradigm consisted the imagination of movement of the left hand (class 1), right hand (class 2), both feet (class 3), and tongue (class 4). Two sessions on different days were recorded for each subject. Each session is comprised of 6 runs separated by short breaks. 
One run consists of $48$ trials ($12$ for each of the four possible classes), yielding a total of $288$ trials per session.

For instance, the data for the first participant are provided in two files labeled as A01T/A01E for the two complete sessions. Each file contains $288$ samples, where 'T' is used for training and 'E' for testing.

The data are collected at a rate of $250 Hz$. The CSV file represents $4$ seconds of data, specifically the data from 2 to 6 seconds in **Figure 1** [2], totaling $1000$ data points. The data underwent bandpass filtering from $0.5$ to $100 Hz$ and a notch filter at $50 Hz$ to eliminate noise interference.

The dataset utilizes the **international 10-20 system** for electrode placement. There are $25$ channels in total, of which three channels are for Electrooculography (EOG) data. For classifying MI (Motor Imagery) signals, it is necessary to filter the channels during preprocessing, retaining $22$ channels. The electrode montage is shown in **Figure 2**.[2]

![alt text](https://github.com/HetuLii/Data-Science-Motor-Imagery-EEG-signal-decoding/blob/main/images/Cue-based%20BCI%20paradigm.png)
*Figure 1: Timing scheme of the cue-based BCI paradigm*
![alt text](https://github.com/HetuLii/Data-Science-Motor-Imagery-EEG-signal-decoding/blob/main/images/Electrode%20montage.png)
*Figure 2: Left: Electrode montage corresponding to the international 10-20 system. Right: Electrode montage of the three monopolar EOG channels*
# Data Processing
## Pre-Processing
Before any processing, each subject gives a dataset with shape $(576, 22000)$. We can impose a butterworth filter ($0.1 Hz$ to $39 Hz$) to reduce noise and irrelevant information. To avoid vanishing/exploding gradient problems, we impose Min-Max Scaling. We then need to expand the dimension such that various models can be trained on the dataset. 
We first reshape the model to the shape $(576, 22, 1000)$, where axis $1$ denotes the number of electrodes (channels), and the last axis denotes the number of samplings per electrode ($4 \cdot 250 = 1000$). This is already sufficient for models such as Long Short-Term Memory (LSTM) and 1D-CNN model. For models involving 2D-CNN, we need to upscale the dataset again to shape $(576, 1, 22, 1000)$.
## Corss-subject dataset
To check the cross-subject performances of the models, we need to choose one subject and take its dataset as the testing dataset, which will include $576$ data points. The rest eight subjects contribute to the training dataset, which thus includes $4608$ data points. Then, we split them into batches with size $32$, which is for parallelism during optimization. 
