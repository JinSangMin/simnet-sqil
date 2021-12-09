from l5kit.data.zarr_utils import zarr_split

zarr_split('scenes/train.zarr/','scenes/', [{'name': 'train10.zarr', 'split_size_GB':0.9}, {'name': 'train90.zarr', 'split_size_GB':-1}])