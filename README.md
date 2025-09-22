# Hybrid-illumination Multiplexed Fourier Ptychographic Microscopy (HMFPM)

This repository contains the implementation and demonstration datasets for the paper:  

**Hybrid-illumination multiplexed Fourier ptychographic microscopy with robust aberration correction**  

arXiv: [https://arxiv.org/abs/2509.05549](https://arxiv.org/abs/2509.05549)

---

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Results](#example-results)
- [Citation](#citation)
- [License](#license)

---

## Introduction
**Hybrid-illumination Multiplexed Fourier Ptychographic Microscopy (HMFPM)** is an advanced computational imaging framework that integrates the advantages of **multiplexed FPM** and **Analytic Pupil function with Incoherent Contrast (APIC)** methods.  

The workflow consists of two stages:

1. **Bright-field initialization**  
   - Acquire eight NA-matched measurements.  
   - Apply **Kramers–Kronig (K–K) relations** and analytic aberration extraction to reconstruct the **bright-field spectrum**.  
   - Estimate the pupil function analytically.  

2. **Dark-field multiplexed reconstruction**  
   - Record a small number of dark-field measurements with **3–6 simultaneously illuminated LEDs** in specially designed multiplexing patterns.  
   - Use a customized optimization algorithm to reconstruct the dark-field spectrum, initialized and constrained by the aberration-corrected bright-field spectrum and the extracted pupil function.
   - 
<p align="center">
  <table>
    <tr>
      <td><img src="./figures/Figure_1.jpg" alt="Experimental setup and illumination strategy for HMFPM" width="350"/></td>
      <td><img src="./figures/Figure_2.jpg" alt="Reconstruction pipeline for HMFPM" width="350"/></td>
    </tr>
    <tr>
      <td align="center"><em>(1) Experimental setup and illumination strategy for HMFPM </em></td>
      <td align="center"><em>(2) Reconstruction pipeline for HMFPM </em></td>
    </tr>
  </table>
  <br>
  <em>Figure 1: HMFPM integrates APIC-based bright-field initialization (a) with multiplexed dark-field reconstruction (b), significantly reducing required measurements while ensuring robust aberration correction.</em>
</p>

**Advantages over MFPM and APIC:**  
- Significantly reduces the number of required measurements.  
- Provides robust aberration correction.  
- Ensures fast and stable convergence without tuning relaxation factors.  

---

## Repository Structure
```bash
├── Data                               # Raw simulation and experimental datasets
├── HMFPM_experiment.py                # Main pipeline for experimental data
├── HMFPM_simulation.py                # Main pipeline for simulation data
├── subfunctionAPIC                    # APIC-related subfunctions for reconstruction
└── README.md                          # Project documentation

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/Magishe/WSI-APIC.git
    ```

2. Navigate to the project directory:
    ```bash
    cd WSI-APIC
    ```

3. Install the dependencies:
   To set up your environment and install all the necessary packages, run the following command:
    ```bash
    pip3 install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torch-dct==0.1.6 --index-url https://download.pytorch.org/whl/cu121
    pip3 install numpy scipy matplotlib pillow h5py opencv-python
    ```


## Usage

### 1. Sample location segmentation
Implement `Sample_Segmentation.py` to automatically locate and segment samples from the image captured by our sample-locating system (`Sample location segmentation/NSCLC.tif`).

    python Sample_Segmentation.py
  

### 2. APIC Reconstruction
Implement `APIC_Reconstruction.py` to perform GPU-accelerated APIC reconstruction on small ROI patches.

    python APIC_Reconstruction.py

Tunable Parameters:
#### (1). Dataloading
Assume we want to reconstruct the Siemens Star sample which was imaged using a highly aberrated imaging system, which is inside a folder named "Data". Then, we modify the code as
      
      python APIC_Reconstruction.py --folderName 'Data'

As there is only one file inside the reducedData folder whose name contains "Siemens_Star_g", we can set ```fileNameKeyword``` with name "Siemens_Star_g". If there are multiple files, then we could use ```additionalKeyword```

      python APIC_Reconstruction.py --folderName 'Data' --fileNameKeyword 'Siemens_Star_g'

#### (2). Basic parameters
1. `enableROI`: When it is set to `false`, the program uses the entire field-of-view in the reconstruction. It is recommended to set to `true` as APIC scales badly with respect to the patch size. A good practice is conducting reconstruction using multiple patches and stiching them together to obtain a larger reconstruction coverage.
2. `ROILength`: This parameter is used only when `useROI` is `true`. It specifies the patch sizes used in the reconstruction. It is preferable to set this to be below 256.
3. `ROIcenter`: Define the center of ROI. Example: ROIcenter = [256,256]; ROIcenter = 'auto'.
4. `useAbeCorrection`: Whether to enable aberration correction. It is always recommended to set to `true`. We keep this parameter so that one can see the influence of the aberration if we do not take aberration into consideration.
5. `paddingHighRes`: To generate a high-resolution image, upsampling is typically requried due to the requirement of Nyquist sampling. `paddingHighRes` tells the program the upsampling ratio.

Demo Usage:

      python APIC_Reconstruction.py --enableROI --ROILength 256 --ROIcenter auto --useAbeCorrection --paddingHighRes 3

### 3. APIC Reconstruction for WholeFOV
GPU-accelerated reconstruction script for full-FOV (2560x2560) images, including auto-stitching functionality

    python APIC_Reconstruction_WholeFOV.py

Tunable Parameters:
1. `patchNumber`: The total number of patches into which you intend to divide the full field of view (FOV) along one dimension
2. `overlappingSize`: The overlap size between different patches (for stitching inside one FOV)

Demo Usage:

      python APIC_Reconstruction_WholeFOV.py --patchNumber 5 --overlappingSize 20

## BiBTeX
@misc{zhao2025hybridilluminationmultiplexedfourierptychographic,
      title={Hybrid-illumination multiplexed Fourier ptychographic microscopy with robust aberration correction}, 
      author={Shi Zhao and Haowen Zhou and Changhuei Yang},
      year={2025},
      eprint={2509.05549},
      archivePrefix={arXiv},
      primaryClass={physics.optics},
      url={https://arxiv.org/abs/2509.05549}, 
}

