# Multimodal Models with Narrow Domain

This project focuses on fine-tuning multimodal models to enhance their performance in specialized, narrow-domain applications. Below are the datasets utilized in this project, categorized by their respective domains.

## Report: 

## Datasets

### Medical Domain Data
For the medical domain, we utilized data from the Human Connectome Project. This dataset provides comprehensive neural imaging data (fMRI), essential for advancing research in brain connectivity and related studies.

- **Human Connectome Project (Young Adult Data)**:  
  [Human Connectome Project - Young Adult Data Releases](https://www.humanconnectome.org/study/hcp-young-adult/data-releases)

### Meme Data
For studying meme-related content, we used the following datasets, which include diverse collections of memes and related metadata. These datasets are crucial for research in social media, cultural analysis, and AI-based meme generation.

- **Meme Captioning Dataset**:  
  [Meme-Cap Dataset on GitHub](https://github.com/eujhwang/meme-cap)

- **MemeCraft Dataset**:  
  [MemeCraft Dataset on GitHub](https://github.com/Social-AI-Studio/MemeCraft.git)

## Methodologies for Medical Domain images 

## Generative modeling

### Bridge Matching
**Bridge Matching** involves training on paired fMRI data, specifically resting-state and motor-task functional connectivity matrices. The key idea is to use conditional diffusion models to align functional connectivity patterns by minimizing the distance between the model output and the target connectivity matrices.

#### Bridge Matching Theory

In conditional diffusion models, we learn to generate samples conditioned on additional information, such as class labels or textual prompts. The training process involves minimizing the loss between the model's prediction and the target using the conditional score functions.

**Training Algorithm**:
1. Generate a pair $(x_0, x_1) \sim q_{01}(x_0, x_1)$.
2. Sample $t \sim \mathcal{U}[0, 1]$.
3. Sample the interpolation $x_t$ from $p_t(x_t | x_0, x_1)$.
4. Feed the pair $(x_t, t)$ into the neural network to compute $f_\theta(x_t, t)$.
5. Compute the conditional vector field $\beta_t \cdot (x_1 - x_t) / \overline{\sigma}^2_t$.
6. Perform a gradient descent step: $\nabla_\theta \left\| f_\theta(x_t, t) - \beta_t \frac{x_1 - x_t}{\overline{\sigma}^2_t} \right\|^2. $

During generation, start with an initial sample $X_0 \sim q_{0}$ and evolve it through an SDE using methods such as Euler's method or other solvers. This approach ensures that the generated samples approximate the target distribution.

### Stable Diffusion 1.5

**Stable Diffusion 1.5** was utilized to enhance the quality of generated samples by conditioning the diffusion process on target images. In our approach, we added motor-task fMRI connectivity matrices as the target, corresponding to the resting-state matrices of the same patient.

**Optimization Approach**:
- We integrated target images into the diffusion process, where the model aimed to minimize the difference between the generated images and the target images using diffusion techniques. 
- The goal was to align the input images to the target images, effectively refining the diffusion process to produce outputs that closely match the desired target.

This method leverages the strengths of Stable Diffusion to achieve high fidelity and accuracy in generating images that correspond to complex connectivity patterns, crucial for analyzing fMRI data in a medical context.

## Methodologies for Memes Domain finetuning

The preparation of training examples included the search and selection of available meme datasets, with a description of what is happening on it. The data was prepared for a format that is compatible with additional finetuning and then the LoRA was used. To test the work of the finetuned model, a competition was made between the basic model and the finetuned one.

## Multimodal models

**GILL Model**:
- **Repository**: [GILL on GitHub](https://github.com/kohjingyu/gill/tree/main?tab=readme-ov-file)
- **Overview**: The GILL model facilitates the integration and analysis of multimodal data, leveraging embeddings and functional connectivity matrices to deliver insightful results.

By preparing this multimodal paired dataset and using the GILL model, we aim to enhance the accuracy and interpretability of fMRI analyses, enabling deeper insights into brain connectivity patterns.

This structured and detailed description should provide a clear overview of your approach and methodologies, highlighting the use of GILL and other techniques in your project.

**TinyLLaVA-Phi-2-SigLIP-3.1B**:
- **Repository**: [TinyLLaVA-Phi-2-SigLIP-3.1B on GitHub]([https://github.com/kohjingyu/gill/tree/main?tab=readme-ov-file](https://github.com/TinyLLaVA/TinyLLaVA_Factory))
- **Overview**: TinyLLaVA Factory is an open-source modular codebase for small-scale large multimodal models (LMMs), implemented in PyTorch and HuggingFace, with a focus on simplicity of code implementations, extensibility of new features, and reproducibility of training results

By preparing multimodal paired dataset and using the TinyLLaVA-Phi-2-SigLIP-3.1B model, we aim to enhance the accuracy and interpretability of memes, enabling a deeper understanding of the context and hidden meaning of the meme.

## Project Overview

This project aims to refine and optimize multimodal models for application in specific domains, leveraging the data from the sources mentioned above. The fine-tuning process is critical for ensuring that these models can accurately interpret and generate content within narrow, specialized contexts, whether in medical research or social media meme analysis.



