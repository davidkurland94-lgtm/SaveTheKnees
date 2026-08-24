# Project Introduction

This project has two separate objectives:

1. [The Kaggle competition](#1-the-kaggle-competition)
2. [The Le Wagon final project](#2-le-wagon-final-project)

---

## 1. The Kaggle Competition

### Goal

Return **one submission** whose score is as close to **1** as possible.

- The submission is a prediction of **confidence scores between 0 and 1**.

### Given

- **569 GB** of dataset
- **4,407 studies** in total, of which **58 are fully labeled**
- Each file records: MRI procedure → patient → picture → radiologist → report

### Setup

Within each study:

- **Unique value** — patient study identification
- **Report** — written in different languages
- **All other columns** relate to the patient's health and to the training set (values between 0 and 1)
  - These series are the X/Y/Z axis views
  - They include fluid/fat sequences

---

## 2. Le Wagon Final Project

### Main product

Using the Kaggle database, build an application that analyses a set of MRI sequences and:

- Returns a **3D view** (3 axes) of the MRI, with problem areas highlighted in color
- Returns a **list of confidence percentages** describing what is happening in the sequence
- Does **not** return a statement or conclusion

### Optional

- **Re-train the model based on the doctor's choice:**
  - The doctor gives a binary validation of the result
  - The validation and report are injected back into the model for training
