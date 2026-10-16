# Beyond the Black Box: an Aggregation Multi-Objective Genetic Approach for Interpretable COVID-19 Diagnosis

This repository contains the official implementation of the paper **"Beyond the Black Box: an Aggregation Multi-Objective Genetic Approach for Interpretable COVID-19 Diagnosis"**.

It proposes an evolutionary framework based on Genetic Algorithms (GAs) to optimize the training hyperparameters of Convolutional Neural Networks (CNNs). The goal is to jointly maximize classification **Accuracy** and medical **Interpretability**, measured through the **Local over Total (LoT)** metric using Grad-CAM attention maps.

## 📖 About the Project

Standard deep learning models for medical imaging are typically trained via gradient updates driven by loss functions that optimize solely for dataset accuracy, often at the expense of transparency.

This project tackles this challenge by framing the training process as a multi-objective optimization problem. We aggregate accuracy $f(x)$ and interpretability $g(x)$ into a single fitness function:

$$
o(x) = \alpha \cdot f(x) + (1-\alpha) \cdot g(x)
$$

Using a dataset of Chest X-Rays for COVID-19 diagnosis, our genetic algorithm explores hyperparameter spaces (Learning Rate, $\beta_1$, $\beta_2$, $\epsilon$, and a Fourier mask radius) to find configurations that force models like AlexNet and VGG11 to focus on clinically valid regions of interest (ROI) rather than non-clinical visual shortcuts.

### Key Findings

* **Interpretability vs. Accuracy Trade-off:** There is a strict trade-off where increasing explainability can reduce raw accuracy. A balanced scalarization weight ($\alpha=0.5$) yielded the most robust Pareto-optimal results.

* **Architecture Matters:** Architectures with a higher proportion of fully connected layers (like AlexNet) exhibit a greater capacity for interpretability improvements (achieving LoT scores up to 0.5823) compared to deeper, convolution-heavy networks like MobileNet or ResNet.

## 🗂 Repository Structure

* `Genetic.py`: Main execution script. Contains the $(\mu+\lambda)$ Genetic Algorithm implementation with elitist selection, crossover, and mutation operators.

* `Fourier_experiment.py`: Trains the CNN model, computes Grad-CAM explanations using `xplique`, evaluates the LoT metric, and writes the fitness to `population.log`.

* `Fourier_preprocess.py`: Applies frequency-domain preprocessing using circular and elliptical Fourier masks.

* `architectures.py`: Contains the TensorFlow/Keras definitions for the evaluated architectures (AlexNet, VGG11, VGG13, VGG16).

* `utils.py`: Helper functions for reading files, images, and printing classification metrics.

* `exp.sh` / `expAlexNet.sh`: Shell scripts containing examples of direct hyperparameter evaluations.

## ⚙️ Installation & Setup

1. **Clone the repository:**

   ```
   git clone https://github.com/your-username/Genetic_XAI.git
   cd Genetic_XAI
   
   ```

2. **Install the dependencies:**
   It is recommended to use a virtual environment or Conda.

   ```
   pip install -r requirements.txt
   
   ```

   *(Required packages include `tensorflow`, `xplique`, `opencv-python`, `scikit-learn`, `numpy`, and `matplotlib`)*

3. **Dataset Preparation:**
   This project relies on a Kaggle COVID-19 Chest X-Ray dataset. You must download the dataset and organize it in the following structure at the root of the project (or update the `--dataset` path):

   ```
   Kaggle_v3/
   ├── COVID/
   │   ├── images/
   │   └── masks/
   ├── Lung_Opacity/
   │   ├── images/
   │   └── masks/
   ├── Normal/
   │   ├── images/
   │   └── masks/
   └── Viral_Pneumonia/
       ├── images/
       └── masks/
   
   ```

   *Note: The `masks/` directories must contain the ground-truth lung segmentations (e.g., generated via U-Net) to compute the LoT metric.*

## 🚀 Usage

To start the hyperparameter optimization process using the Genetic Algorithm, simply run:

```
python Genetic.py

```

The script will automatically instantiate the initial population, call `Fourier_experiment.py` as a subprocess to train and evaluate each model, and manage the generations. Results are appended to `population.log`.

**Important considerations:**

* Ensure you create a folder named `explanaitions_VGG_11/` in the root directory before running if you want to save the Grad-CAM visualization outputs, as the script writes to this path.

* Training multiple CNNs via genetic algorithms is computationally intensive. The experiments in the paper were run on a workstation with dual NVIDIA RTX 3090 GPUs.

## 📊 Results Summary

The following table presents the best hyperparameter configurations identified by the evolutionary framework ($\alpha=0.5$):

| Model | LR | $\beta_1$ | $\beta_2$ | $\epsilon$ | LoT (Interpretability) | Accuracy | Aggregation Fitness | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **MobileNet** | 0.00062 | 0.87 | 0.19 | 5.4e-7 | 0.2224 | **0.9246** | 0.5735 | 
| **VGG11** | 7e-05 | 0.22 | 0.98 | 8.0e-7 | 0.4216 | 0.9005 | 0.6611 | 
| **AlexNet** | 0.00030 | 0.48 | 0.23 | 6.4e-7 | **0.5823** | 0.8497 | **0.7160** | 
| **ResNet** | 0.00016 | 0.62 | 0.05 | 6.6e-7 | 0.4734 | 0.8785 | 0.6760 | 

## 📝 Citation

If you use this code or our methodology in your research, please consider citing our paper:

```
@article{BeyondTheBlackBox2026,
  title={Beyond the Black Box: an Aggregation Multi-Objective Genetic Approach for Interpretable COVID-19 Diagnosis},
  author={[Your Name/Authors]},
  journal={[Journal/Conference Name]},
  year={2026}
}

```

## 📄 License

\[Specify your license here, e.g., MIT License\]