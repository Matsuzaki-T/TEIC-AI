# TEIC-AI

## Overview
This repository contains the code an the dataset used for our manuscript. TEIC-AI is a predictive model that operates on tailoring initial dosing regimen of teicoplanin, a glycopeptide antibiotic with activity against Gram-positive bacteria including methicillin-resistant Staphylococcus aureus (MRSA).


## Software requirements
- Python >=3.9.12
- NumPy >=1.21.5
- Pandas >=1.4.2
- PsmPy >= 0.3.5
- matplotlib >= 3.5.1
- seaborn >= 0.11.2
- scikit-larn >=1.0.2

## Quick start
After cloning the repository, open the Demo.ipynb file in your cloned directory, enter the patient' data, and then run the code to tailor dose.
```
import TEIC_AI

#enter the following parameters
Age = ""
BW =""#body weight (kg)
BMI = ""
Ccr = "" #creatinine clearance(mL/min)
Alb = "" #serun albumin(g/dL)
T0 = "" #enter the time (Only integer. e.g. 8:51 to 8 and 13:43 to 13)
T1 = 0 #change T1 to 1 (Monday)
T2 = 0 #change T2 to 1 (Tuesday)
T3 = 0 #change T3 to 1 (Wednesday)
T4 = 0 #change T4 to 1 (Thursday)
T5 = 0 #change T5 to 1 (Friday)
T6 = 0 #change T6 to 1 (Saturday)
T7 = 0 #change T7 to 1 (Sunday)

TEIC_AI.planning(Age,BW,BMI,Ccr,Alb,T0,T1,T2,T3,T4,T5,T6,T7)

```
