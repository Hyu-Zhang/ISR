# ISR

Official Implementation of "Uncovering Hidden Connections: Iterative Search and Reasoning for Video-grounded Dialog".

![](framework.jpg)

## Abstract
In contrast to conventional visual question answering, video-grounded dialog necessitates a profound understanding of both dialog history and video content for accurate response generation. Despite commendable progress made by existing approaches, they still face the challenges of incrementally understanding complex dialog history and assimilating video information. In response to these challenges, we present an iterative search and reasoning framework, which consists of a textual encoder, a visual encoder, and a generator. Specifically, we devise a path search and aggregation strategy in textual encoder, mining core cues from dialog history that are pivotal to understanding the posed questions. Concurrently, our visual encoder harnesses an iterative reasoning network to extract and emphasize critical visual markers from videos, enhancing the depth of visual comprehension. Finally, we utilize the pre-trained GPT-2 model as our answer generator to decode the mined hidden clues into coherent and contextualized answers. Extensive experiments on three public datasets demonstrate the effectiveness and generalizability of our proposed framework.

## Data

You can visit this [link](https://drive.google.com/drive/folders/1SlZTySJAk_2tiMG5F8ivxCfOl_OWwd_Q) to get the text data, extracted video/audio feature, and evaluation tools.

If you need the raw video data, you can access this [website](https://prior.allenai.org/projects/charades). And the original videos of the test sets of the official Charades Challenge can be downloaded from the below updated links:  

https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_vu17_test.tar [13GB]  
https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_vu17_test_480.tar [2GB]

## Code

We have released the code used. 
