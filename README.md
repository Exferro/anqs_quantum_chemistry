# Autoregressive neural quantum states for quantum chemistry 
Supporting code for the papers "Autoregressive neural quantum states with quantum number symmetries" and "Neural quantum states and peaked molecular wave functions: curse or blessing?".

At the moment this repository is under the active development, stay tuned for more information.

# Quick start
A dedicated Google Colab notebook (see the link below) provides a toy example of some simple calculations for a Li2O molecule with 30 qubits. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Exferro/anqs_quantum_chemistry/blob/main/colab_toy_model.ipynb)

# Installation
Depending on whether you have CUDA installed or not, as well as depending on the CUDA version we offer three possible installation pathways. 
All pathways suppose downloading the code contained in this repository and running the following commands **after** changing the working directory to the directory of this repository (e.g. being inside the `anqs_quantum_chemistry` directory)

## CPU version
```bash
pip3 install --extra-index-url https://download.pytorch.org/whl/cpu . anqs
```

## CUDA 11 version
```bash
pip3 install --extra-index-url https://download.pytorch.org/whl/cu118 . anqs[cuda11]
```

## CUDA 12 version
```bash
pip3 install . anqs[cuda12]
```