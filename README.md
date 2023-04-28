# wild_places_processing
repo to process the Wild Places dataset into my data pipeline.

At the moment, there are two scripts. `scripts/register_pointclouds.py` grabs pointclouds from the dataset and registers them together. It then extracts BEV-space map features and an open3d pointcloud for visualization. `scripts/convert_to_rosbag.py` creates a rosbag out of a run in the dataset.
