# NuiSI Dataset

**Nui**track **S**keleton **I**nteraction Dataset of Physical Human-Human Interactions

## Extracting the trajectories

The trajectories were manually inspected and the starting and stop points were identified. The scripts [`preproc_v1.py`](preproc_v1.py) and [`preproc_v2.py`](preproc_v2.py) can be run to preprocess the data of versions 1 and 2 of the dataset respectively. They will process the data to give N trajectories of size Tx6M where T is the (varying) length of each trajectory and M is the number of joints, 
The first half of the dimensions, until 3M are the (flattened) 3D coordinates of the first agent and the second half are the 3D coordinates of the second agent.

The trajectory data is saved with the keys `train_data` and `test_data` along with a trajectory-wise (*not frame-wise*) label as `train_labels` and `test_labels` in a compressed `.npz` file. See [`visualize.py`](visualize.py) for further clarification.

The actions in v1 are:

- Clapfist: Clapping the hands horizontally followed by a fistbump
- Fistbump: A normal fistbump
- Handshake: A normal handshake
- Highfive: A normal highfive
- Rocket Fistbump: Both persons bump their fists at a waist level and then synchronously move them upwards like a rocket, while maintaining contact
- Wave: Waving at each other

The actions in v2 are the same as those in [Bütepage et al (2021) "Imitating by Generating: Deep Generative Models for Imitation of Interactive Tasks"](https://github.com/jbutepage/human_robot_interaction_data):

- Waving: Waving at each other
- Handshake: A normal handshake
- Rocket Fistbump: Both persons bump their fists at a waist level and then synchronously move them upwards like a rocket, while keeping contact
- Parachute Fistbump: Both persons bump their fists at a chest/shoulder level and then synchronously oscillate them sideways while coming down like a parachute, while maintaining contact

## Citation

If you used this for your work, please consider citing us to spread the love:

```latex
@inproceedings{prasad2022mild,
  title={MILD: Multimodal Interactive Latent Dynamics for Learning Human-Robot Interaction},
  author={Prasad, Vignesh and Koert, Dorothea and Stock-Homburg, Ruth and Peters, Jan and Chalvatzaki, Georgia},
  booktitle={IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  year={2022}
}
```

## Acknowledgement

The authors are thankful to [Louis Sterker](https://github.com/enilois), [Erik Prescher](https://github.com/ErikPre) and [Sven Schultze](https://github.com/svenschultze) whose help resulted in this dataset.
