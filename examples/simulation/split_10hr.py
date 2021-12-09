from l5kit.data.zarr_utils import zarr_split

zarr_split('/data2/minji/dataset/l5kit_data/scenes/train.zarr/','/data2/minji/dataset/l5kit_data/scenes/', [{'name': 'train10.zarr', 'split_size_GB':0.9}, {'name': 'train90.zarr', 'split_size_GB':-1}])