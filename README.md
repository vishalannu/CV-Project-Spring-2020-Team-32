StoryGraphs : Visualizing Character Interactions as a Timeline  
===========

This is a python implementation of the paper:

[Paper download](https://cvhci.anthropomatik.kit.edu/~mtapaswi/papers/CVPR2014.pdf) 

----

### Main functions
- [scene_detection.py](scene_detection/scene_detection.py)   Generates the different scenes prensent in the episode.
- [shot_boundary_detection.py](shot_detection/shot_boundary_detection.py)   Generates shots boundaries to differentiate different shots of the episode.
- [shot_hist_representation.py](shot_representation/shot_hist_representation.py)   Represntaion of shots using histogram.
- [shot_threading.py](shot_threading/shot_threading.py)   Connects all the shots and generates threads of shots in a scene.

