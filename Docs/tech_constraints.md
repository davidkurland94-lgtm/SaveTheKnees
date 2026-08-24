Kaggle constraints:




Wagon constraints:

**PROCESS**:
Most rows in train.csv have reports but no result on other columns butthey have the images sequence,
so we need to create models that can exclusively observe/train on images.
As they have reports, we also need to train models on reports only, in order to get the words that are used to describe the patient knee status.
Once these results


- 1 model trained on images:
-> train : train the model through the kaggle dataset
-> purpose: The model should be able to compare 1 new image to the previous train set
-> result: the new image of the knee : is healthy ? has anomalies ?
-> we should have a correlation between 3 close images from a sequence (eq: if a knee looks injured on an image, there should be high prob that the injure appears in img-1 and img+1 of the sequence).


compiler:
the model should accept a set from img0 to img45 (depending of nb of images per set) in order to create an axis view
the model should then compile 3 sets of images -> X/Y and Z in order to get a 3D view of the MRI

- 2 model trained on reports:
-> convert reports in english

Question:
How to figure out if the model considers a baby / old man's knee ?
How shall we handle outliers ? Shall we start with a Central Limit Theorem to check confidence and outliers ?


feedback:
take it simple ¥
-> focus on simple
-> volume rendering framework

