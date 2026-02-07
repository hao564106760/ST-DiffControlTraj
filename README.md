# ST-DiffControlTraj

1) Environment
--------------
Tested on:
- OS: Linux (x86_64)
- Python: 3.7.4
- GPU: NVIDIA RTX A5000
- CUDA (torch): 11.7

Main packages:
- torch 1.13.1+cu117
- torchvision 0.14.1+cu117
- numpy 1.21.6
- scipy 1.7.3
- matplotlib 3.5.3
- tqdm 4.36.1
- osmnx 1.1.2
- networkx 2.6.3

Optional (install if needed by your scripts):
- haversine
- colored

2) Data
-------
Download the dataset(s) and place files under ./data/ .
This repo does not include raw data. Please follow the dataset license/terms.

3) Preprocessing
----------------
Run your trajectory preprocessing to generate:
- processed trajectories (e.g., *.npy)
- dataset stats (e.g., *.npz for normalization)

Road network preprocessing (RoadMAE):
- Build road graph features from GraphML (produces porto_roadmae_data.pt or similar)
- Train RoadMAE to export road embeddings (road_embeddings.pt)

Make sure all output paths match utils/config.py and the embedding path used in the model.

4) Training
-----------
Edit paths / hyperparameters in utils/config.py, then run:
    python main.py

Outputs are saved under ./ST-DiffControlTraj/ (models, logs, and results).
