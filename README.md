# Universal Function Approximation via Neural Networks

## 📊 Project Overview

The goal of this project is to demonstrate the **Universal Approximation Theorem** by constructing a Neural Network from scratch. This implementation proves that a feedforward neural network with a single hidden layer can approximate any continuous function (such as $\sin(x)$ or $x^2$) given sufficient neurons and appropriate training.

The project demonstrates the mathematics behind neural networks without the abstraction of high-level libraries like PyTorch or TensorFlow.

---

## 🧠 Theoretical Background

### The Universal Approximation Theorem

The Universal Approximation Theorem states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of $\mathbb{R}^n$, under certain assumptions on the activation function.

Formally, let $\sigma(z)$ be a non-constant, bounded, and continuous activation function. Let $I_m$ denote the $m$-dimensional unit hypercube $[0, 1]^m$. The space of continuous functions on $I_m$ is denoted by $C(I_m)$.

Given any function $f \in C(I_m)$ and error tolerance $\varepsilon > 0$, there exists an integer $N$ (neurons) and parameters $v_i, b_i, w_i$ such that the function $F(x)$:

$$
F(x) = \sum_{i=1}^{N} v_i \sigma(w_i^T x + b_i)
$$

satisfies:

$$
| F(x) - f(x) | < \varepsilon \quad \forall x \in I_m
$$

### Activation Functions

Non-linearity is crucial for the network to learn complex patterns. Three primary activation functions are utilized. The choice of activation affects the gradient flow during backpropagation.

| Function | Formula $\sigma(z)$ | Derivative $\sigma'(z)$ | Characteristics                              |
| :--- | :--- | :--- |:---------------------------------------------|
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | Smooth, bounded $(0, 1)$.                    |
| **Tanh** | $\tanh(z)$ | $1 - \tanh^2(z)$ | Smooth, bounded $(-1, 1)$.                   |
| **ReLU** | $\max(0, z)$ | $1 \text{ if } z > 0, \text{ else } 0$ | Efficient. Unbounded output.                 |

### Forward Propagation

The network follows a standard 2-layer architecture:

1.  **Hidden Layer:**
    $$Z^{[1]} = X \cdot W^{[1]} + b^{[1]}$$
    $$A^{[1]} = \sigma(Z^{[1]})$$
2.  **Output Layer:**
    $$Z^{[2]} = A^{[1]} \cdot W^{[2]} + b^{[2]}$$
    $$\hat{Y} = Z^{[2]}$$

### Backpropagation & The Chain Rule

To train the network, we minimize the **Mean Squared Error (MSE)** loss function $J$. We compute the gradient of $J$ with respect to the weights using the Chain Rule.

**Step A: Error at Output**<br>
First, we calculate the derivative of the loss with respect to the output layer's input ($Z^{[2]}$). Since the output activation is linear for regression:
$$\delta^{[2]} = \frac{\partial J}{\partial Z^{[2]}} = (\hat{Y} - Y)$$

**Step B: Propagating to Hidden Layer**<br>
We propagate the error backwards to the hidden layer. This requires the **Hadamard product** ($\odot$), which is element-wise multiplication, to apply the derivative of the activation function:

$$
\delta^{[1]} = (\delta^{[2]} \cdot W^{[2]T}) \odot \sigma'(Z^{[1]})
$$

**Step C: Gradients for Updates**<br>
Finally, we calculate the gradients for the weights and biases:

$$
\frac{\partial J}{\partial W^{[2]}} = A^{[1]T} \cdot \delta^{[2]}
$$

$$
\frac{\partial J}{\partial W^{[1]}} = X^T \cdot \delta^{[1]}
$$

---

## 📉 Optimization Methods

This repository contains three separate Jupyter Notebooks, each implementing the approximation using a different optimization strategy. The specific mathematical update rules for each method are detailed within the respective files.

| Notebook | Method | Description |
| :--- | :--- | :--- |
| `BGD.ipynb` | **Batch Gradient Descent** | Uses the **entire dataset** to calculate the gradient for a single update step. Stable but computationally expensive. |
| `SGD.ipynb` | **Stochastic Gradient Descent** | Uses a **single random sample** for each update. Highly noisy but converges faster per epoch. |
| `MBGD.ipynb`| **Mini-Batch Gradient Descent** | Uses a **subset of samples** (e.g., 32 or 64) per update. Balances the stability of BGD with the speed of SGD. |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mattia-3rne/Universal-Function-Approximation-via-Neural-Networks.git
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Analysis

1.  Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

2.  Open any of the three notebooks (`BGD.ipynb`, `SGD.ipynb`, or `MBGD.ipynb`).

3.  **Run All Cells**:
    The notebooks are self-contained. You can modify the `target_function` variable in the configuration cell to change the function being approximated (e.g., change `np.sin(x)` to `x**2`).

---

## 📂 Project Structure

* `requirements.txt`: Python package dependencies.
* `MBGD.ipynb`: Mini-Batch Gradient Descent.
* `SGD.ipynb`: Stochastic Gradient Descent.
* `BGD.ipynb`: Batch Gradient Descent.
* `README.md`: Project documentation.