# A Data-Centric Approach to Detecting and Mitigating Demographic Bias in Pediatric Mental Health Text

This repository contains the official code accompanying the paper:

### **📄 A Data-Centric Approach to Detecting and Mitigating Demographic Bias in Pediatric Mental Health Text**  
**arXiv (2025): https://arxiv.org/abs/2501.00129**

The project proposes a **data-centric framework** for detecting and mitigating demographic bias in pediatric mental health classification models, focusing on biases arising from **non-biological differences** in clinical documentation.  
All code in this repository operates on *synthetic data*.

---

## 🔬 Overview

This work demonstrates that:

- Female adolescents are **systematically under-diagnosed** using Transformer-based anxiety prediction models.  
- Clinical notes differ substantially in **length**, **lexical density**, and **documentation style** across sex groups.  
- These differences lead to **higher false-negative rates** and inconsistent model performance.  
- A **data-centric debiasing pipeline** based on sentence-level importance filtering and linguistic neutralization—can reduce bias.

The repository provides notebooks:

- **`Debias_Notes.ipynb`** — Implements our debiasing framework.  
- **`Patient_Note_Classifier.ipynb`** — Runs model training, evaluation, and fairness assessment.
- **`get_ber.py`** — script-style implementation to compute BER scores    
- **`fake_notescsv** — Synthetic clinical note.

---

## 📘 Notebooks

### **1. Debias_Notes.ipynb**
Implements:

- TF–IDF sentence importance filtering  
- Randomized length normalization  
- Stanza-based PERSON-name detection  
- Gender neutralization  
- Export of debiased datasets  

---

### **2. Patient_Note_Classifier.ipynb**
Implements:

- Patient-level note aggregation  
- BigBird token-length estimation  
- BigBird + Logistic Regression classification  
- Sex/ethnicity fairness evaluation  
- Visualizations

### **3. get_ber.py** 

Computes Balanced Error Rate (BER) for two demographic subgroups and reports the BER ratio to quantify fairness disparities. It resamples groups to equal size and evaluates whether one group experiences systematically higher error rates.

---

## 📚 Citation

```bibtex
@ARTICLE{Ive2024-fv,
  title         = "A data-centric approach to detecting and mitigating
                   demographic bias in pediatric mental health text: A case
                   study in anxiety detection",
  author        = "Ive, Julia and Bondaronek, Paulina and Yadav, Vishal and
                   Santel, Daniel and Glauser, Tracy and Cheng, Tina and Strawn,
                   Jeffrey R and Agasthya, Greeshma and Tschida, Jordan and
                   Choo, Sanghyun and Chandrashekar, Mayanka and Kapadia, Anuj J
                   and Pestian, John",
  journal       = "arXiv [cs.CL]",
  month         =  dec,
  year          =  2024,
  archivePrefix = "arXiv",
  primaryClass  = "cs.CL"
}

```
