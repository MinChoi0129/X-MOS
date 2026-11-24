from pathlib import Path


def sequence_dataloaders():
    """Name of every dataloader. All of them need a sequence, for HeLiMOS `<sensor>/<split file>`."""
    return ["helimos", "kitti", "apollo"]


def dataset_factory(dataloader: str, data_dir: Path, *args, **kwargs):
    # Imported lazily, the KITTI loader needs the C++ extension of KISS-ICP
    if dataloader == "helimos":
        from datasets.helimos import HeliMOSDataset as Dataset
    elif dataloader == "kitti":
        from datasets.kitti import SemanticKITTIDataset as Dataset
    elif dataloader == "apollo":
        from datasets.apollo import SemanticApolloDataset as Dataset
    else:
        raise ValueError(f"Unknown dataloader {dataloader}, supported are {sequence_dataloaders()}")
    return Dataset(data_dir=data_dir, *args, **kwargs)
