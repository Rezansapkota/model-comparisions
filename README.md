# Comparing machine learning models for IoT traffic

PRT565 Group 16: Raman Lamichhane (s401547), Rejan Sapkota (s400142), and Saroj Dhakal (s399846), Darwin campus.

This project compares Decision Tree, Random Forest, Gaussian Naive Bayes, and a multilayer neural network on RT-IoT2022. Model selection uses validation macro F1. Identical predictor vectors are kept in the same partition to reduce overlap between training and testing.

## Files

- `Model_Comparison.ipynb`: fully executed notebook with results and charts.
- `Model_Comparison.py`: the same code exported as a Python script.
- `data/dataset.csv`: the supplied dataset, unchanged.
- `requirements.txt`: tested direct dependencies.
- `requirements-lock-windows.txt`: complete dependency versions from the tested Windows environment.
- `Setup_Environment.ps1` and `Open_Notebook.cmd`: Windows setup and launcher.

## Run on Windows

Install Python 3.12, then open PowerShell in this repository and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Setup_Environment.ps1
```

After setup, double-click `Open_Notebook.cmd`. In JupyterLab, select **Kernel > Restart Kernel and Run All Cells**, then save the notebook when all cells finish. Allow several minutes for training; no GPU is required.

## Run on other systems

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab Model_Comparison.ipynb
```

Run from the repository root so `data/dataset.csv` is found. Use `requirements.txt` on other systems; the full Windows lock file can contain platform-specific dependencies.

The notebook exports metrics, predictions, figures, split membership, and fitted models into `outputs/`. That directory and virtual environments are excluded from Git.

## Verification

All 12 code cells completed without errors in Python 3.12.13 with TensorFlow 2.21.0 and scikit-learn 1.9.1. Final test scores match the earlier project results at two decimal places:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Decision Tree | 99.79% | 97.05% |
| Random Forest | 99.84% | 97.84% |
| Gaussian Naive Bayes | 92.92% | 65.19% |
| Neural network | 99.59% | 95.93% |

Decision Tree was selected using validation macro F1; Random Forest obtained the highest final test macro F1. These are within-dataset results, not evidence of performance on unseen networks or attacks. Numerical results and training times can vary across software and hardware environments.

## Dataset attribution

S., B., & Nagapadma, R. (2023). *RT-IoT2022* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P338

The dataset is distributed under CC BY 4.0. Its original 123,117 records become 117,922 after removing exact duplicates and excluding the exported index from duplicate checking.
