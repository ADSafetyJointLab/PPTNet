# PPTNet: A Hybrid Periodic Pattern-Transformer Architecture for Traffic Flow Prediction and Congestion Identification

<p align="center">
  <img src="assets/PPTNet.png" alt="PPTNet" />
</p>

---

## Introduction
Accurate prediction of traffic flow parameters and real-time identification of congestion states are essential for the efficient operation of intelligent transportation systems. This paper proposes a Periodic Pattern-Transformer Network (PPTNet) for traffic flow prediction, integrating periodic pattern extraction with the Transformer architecture, coupled with a fuzzy inference method for real-time congestion identification. Firstly, a high-precision traffic flow dataset (Traffic Flow Dataset for China’s Congested Highways \& Expressways, TF4CHE) suitable for congested highway scenarios in China is constructed based on drone aerial imagery data. Subsequently, the proposed PPTNet employs Fast Fourier Transform to capture multi-scale periodic patterns and utilizes two-dimensional Inception convolutions to efficiently extract intra and inter periodic features. A Transformer decoder dynamically models temporal dependencies, enabling accurate predictions of traffic density and speed. Finally, congestion probabilities are calculated in real-time using the predicted outcomes via a Mamdani fuzzy inference-based congestion identification module. Experimental results demonstrate that the proposed PPTNet significantly outperforms mainstream traffic prediction methods in prediction accuracy, and the congestion identification module effectively identifies real-time road congestion states, verifying the superiority and practicality of the proposed method in real-world traffic scenarios.

<p align="center">
  <img src="assets/pipline compare.png" alt="pipline" />
</p>

---

## Environments

- python 3.9, pytorch 2.5.1,  CUDA 12.4

```python
git clone https://github.com/ADSafetyJointLab/PPTNet.git
conda create -n pptnet python=3.9.20
conda activate pptnet
pip install -r requirements.txt
```
---
## Dataset
The **TF4CHE** (Traffic Flow Dataset for China’s Congested Highways & Expressways) is derived from the UAV-based **AD4CHE** dataset, calibrated to approximately 5 cm accuracy at 100 m altitude. TF4CHE is pre-processed into a consolidated time-series format suitable for traffic flow prediction and congestion identification.

- **Coverage**  
  - **11 road segments** (`Road_segment_1.csv` … `Road_segment_11.csv`), each corresponding to a distinct expressway section in five Chinese cities (originally 68 AD4CHE segments, consolidated by route).  
  - **Lane layout:** Four travel lanes + emergency lane in each direction; world coordinate origin at top-left of UAV frame, X increasing in travel direction, Y downward.

- **File format**  
  Each `Road_segment_{i}.csv` contains one row per second, with the following columns:

| Name               | Description                                                                               | Unit      |
|--------------------|-------------------------------------------------------------------------------------------|-----------|
| Month/Year         | Month and year of video recording (virtual date information)                              | –         |
| Weekday            | Completion date of video recording (virtual date information)                             | –         |
| TimeCode           | Specific start time of video recording (virtual time information)                         | –         |
| second             | Video time sequence in seconds                                                            | s         |
| drivingDirection   | Traveling direction of the recorded segment                                               | –         |
| car                | Number of cars in frame                                                                   | veh       |
| bus                | Number of buses in frame                                                                  | veh       |
| truck              | Number of trucks in frame                                                                 | veh       |
| G(t)               | Equivalent vehicle count, with cars as the reference, (conversion coefficients: $\alpha_{\text{bus}} = 2,\ \alpha_{\text{truck}} = 2.5$)     | veh       |
| k(t)               | Average density                                                                            | veh/m     |
| q(t)               | Average flow                                                                               | veh/s     |
| xVelocity(t)       | Mean speed along X-axis                                                                    | m/s       |
| yVelocity(t)       | Mean speed along Y-axis                                                                    | m/s       |
| xAcceleration(t)   | Mean acceleration along X-axis                                                             | m/s²      |
| yAcceleration(t)   | Mean acceleration along Y-axis                                                             | m/s²      |
| OccupancyRatio     | Lane space occupancy                                                                       | –         |
| File_ID            | Original AD4CHE segment index                                                              | –         |


> **Note:** TF4CHE converts per-frame trajectory data (`xx_tracks.csv`) and metadata (`xx_recordingMeta.csv`, `xx_tracksMeta.csv`) into uniformly spaced time-series, applying conversion formulas from rail transit theory to compute densities, flows, and occupancy, thus streamlining downstream forecasting tasks.


- **Downloads**
  We provide download link from Google Drive and Baidu Yunpan to facilate users from all over the world.
  - **[Baidu Yunpan](https://pan.baidu.com/s/170RxFKCbPo0PCzDMVRyFNw?pwd=xrvr)**&emsp;Extraction code&nbsp;:&nbsp;`xrvr`
  - **[Google Drive](https://drive.google.com/drive/folders/18WgRAeoeCMSkDBpFw9REE17ISUBAf2qC?usp=sharing)**

<p align="center">
  <img src="assets/TF4CHE construction.png" alt="TF4CHE" />
</p>

---





