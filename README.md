# Additive-Manufacturing-Sensor-Selection
Optimizing In-situ Monitoring for Laser Powder Bed Fusion Process: Deciphering Acoustic Emission and Sensor Sensitivity with Explainable Machine Learning


# Journal link
https://doi.org/10.1016/j.jmatprotec.2023.118144

# Overview

Metal-based Laser Powder Bed Fusion process (LPBF) has easily facilitated the fabrication of intricately shaped components. But assessing part density is inefficient and relies on costly Computed Tomography (CT) or time-consuming destructive analyses. Inspecting every layer or intermittently hampers machine productivity. To address these challenges, the Additive Manufacturing (AM) field explores real-time monitoring of build quality using sensor signatures and Machine Learning (ML) techniques. One such approach is sensing airborne Acoustic Emissions (AE) from process zone perturbations and comprehending flaw formation for monitoring the LPBF process. This study emphasizes the importance of selecting airborne AE sensors for accurately classifying LPBF dynamics in 316L, utilizing a flat response sensor to capture AE’s during three regimes: Lack of Fusion (LoF), conduction mode, and keyhole pores. To obtain a comprehensive understanding of AE from a broad process space, the data was collected for two different 316L stainless steel powder distributions (> 45 µm and < 45 µm) using two different parameter sets. Frequency domain analysis revealed statistically differentiated LPBF dynamics with dominant frequencies in specific ranges. Empirical Mode Decomposition (EMD) was used to examine the periodicity of AE signals by separating them into constituent signals for comparison. Machine learning classifiers (Convolutional Neural Networks, eXtreme Gradient Boosting, and Support Vector Machines) were trained on transformed AE signals to distinguish regimes. Sensitivity analysis using saliency maps and feature importance scores identified relevant frequency information for decision-making that is below 40 kHz. The study highlights the potential of interpretable machine learning frameworks for identifying crucial frequency ranges in distinguishing LPBF regimes and emphasizes the importance of sensor selection in LPBF process monitoring.  
![Graphical abstract](https://github.com/vigneashpandiyan/Additive-Manufacturing-Sensor-Selection-Acoustic-Emission/assets/39007209/40470ccb-a958-4466-a6f7-a777eb0bd38c)



# Methodology
In this study, we developed two strategies to compute crucial frequency range importance for classification in LPBF regimes. These strategies serve distinct purposes and involve varied preprocessing levels, leading to different input configurations. The first strategy employs a 1D CNN to compute saliency over Intrinsic Mode Functions (IMFs) derived using EMD. This approach requires minimal preprocessing, and the CNN computes feature importance via saliency maps without additional steps. The input for this strategy is set at 7, corresponding to the 7 IMFs obtained through EMD, each carrying specific frequency information. The CNN's inherent capabilities facilitate this process seamlessly. In contrast,  the second strategy involves a more complex process with additional preprocessing steps. This method entails feature computation, followed by using classifiers like XGBoost and SVM to evaluate classification accuracy. Subsequently, the feature importance and SHAP method are applied to the trained model to discern the importance score. The frequency information acquired from manual feature extraction is categorized into 15 bins. The feature computation was done with the periodogram method to compute energies for fifteen frequency bands from 0 to 150 kHz across all dataset windows. This analysis also will provide insight into the AE waveform signal in the frequency domain. 

These two strategies are distinct and serve different analytical purposes. While the first strategy leverages the inherent capabilities of a 1D CNN to compute saliency over IMFs, the second strategy involves a multi-step process incorporating classifiers with feature importance scores and SHAP analysis to assess feature importance. Given the fundamentally different nature of these strategies, a direct comparison between them may not be an ideal approach. Each strategy was designed to provide insights into sensor frequency dependence for decision-making from a unique perspective. Both strategies were validated through comprehensive experimentation and analysis on four different AE datasets. 

![image](https://github.com/vigneashpandiyan/Additive-Manufacturing-Transfer-Learning/assets/39007209/de11305c-119f-4269-b271-8a4847f59e1c)


# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Sensor-Selection-Acoustic-Emission
cd Additive-Manufacturing-Sensor-Selection-Acoustic-Emission
python Main.py
```

# Citation
```
@article{PANDIYAN2023118144,
title = {Optimizing In-situ Monitoring for Laser Powder Bed Fusion Process: Deciphering Acoustic Emission and Sensor Sensitivity with Explainable Machine Learning},
journal = {Journal of Materials Processing Technology},
pages = {118144},
year = {2023},
issn = {0924-0136},
doi = {https://doi.org/10.1016/j.jmatprotec.2023.118144},
url = {https://www.sciencedirect.com/science/article/pii/S0924013623002893},
author = {Vigneashwara Pandiyan and Rafał Wróbel and Christian Leinenbach and Sergey Shevchik},
keywords = {Laser Powder Bed Fusion, Process Monitoring, Empirical Mode Decomposition, Acoustic Emission, Explainable AI (XAI)},
}
