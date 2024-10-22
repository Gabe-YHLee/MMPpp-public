from torch.utils import data

from loader.Toy_dataset import Toy
from loader.Robot_dataset import Robot
from loader.Pouring_dataset import Pouring, HandMadePouring

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict)
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True)
    )
    return loader

def get_dataset(data_dict):
    name = data_dict["dataset"]
    if name == "Toy":
        dataset = Toy(**data_dict)
    elif name == "Robot":
        dataset = Robot(**data_dict)
    elif name == "Pouring":
        dataset = Pouring(**data_dict)
    elif name == "HandMadePouring":
        dataset = HandMadePouring(**data_dict)
    return dataset