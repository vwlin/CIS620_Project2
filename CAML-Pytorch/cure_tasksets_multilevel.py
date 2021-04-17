"""
    Below was referenced from:
    [1] https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/benchmarks/omniglot_benchmark.py
    [2] https://github.com/olivesgatech/CURE-TSR/blob/master/train.py (lines 58-65 in the link) (curetsr dataset loading here)
    Goal: Load task specific data for train, val, test data
"""
import random
import learn2learn as l2l

from torchvision import transforms, datasets
from PIL.Image import LANCZOS
import utils


def cure_tsr_multilevel_intermediate_tasksets(
    inter_dir_list,
    train_ways,
    train_samples,
    test_ways,
    test_samples,
    **kwargs
):
    """
    Benchmark definition for CURE TSR.
    """
    data_transforms = transforms.Compose([transforms.Resize([28, 28]), transforms.ToTensor(), utils.l2normalize, utils.standardization])

    #lvl0_train_dir = './CURE_TSR_OG/Real_Train/ChallengeFree/'
    #lvl5_test_dir = './CURE_TSR_OG/Real_Train/Snow-5/'
    #curetsr_lvl0 = utils.CURETSRDataset(lvl0_train_dir, data_transforms)
    #curetsr_lvl5 = utils.CURETSRDataset(lvl5_test_dir, data_transforms)

    levels = []*len(inter_dir_list)
    for i in range(len(levels)):
        train_dir = inter_dir_list[i]
        curetsr_data = datasets.ImageFolder(train_dir, transform=data_transforms)
        levels[i] = l2l.data.MetaDataset(curetsr_data)
    meta_curetsr_multilevel = l2l.data.UnionMetaDataset(levels)

    train_dataset = meta_curetsr_multilevel
    validation_dataset = meta_curetsr_multilevel
    
    classes = list(range(14)) # 14 classes of stop signs
    random.shuffle(classes)
    train_transforms = [
        l2l.data.transforms.FusedNWaysKShots(train_dataset,
                                             n=train_ways,
                                             k=train_samples,
                                             filter_labels=classes[:8]), # first few classes for training
        l2l.data.transforms.LoadData(train_dataset),
        l2l.data.transforms.RemapLabels(train_dataset),
        l2l.data.transforms.ConsecutiveLabels(train_dataset),
    ]
    validation_transforms = [
        l2l.data.transforms.FusedNWaysKShots(validation_dataset,
                                             n=test_ways,
                                             k=test_samples,
                                             filter_labels=classes[8:14]), # last few classes for val / test
        l2l.data.transforms.LoadData(validation_dataset),
        l2l.data.transforms.RemapLabels(validation_dataset),
        l2l.data.transforms.ConsecutiveLabels(validation_dataset),
    ]

    _datasets = (train_dataset, validation_dataset)
    _transforms = (train_transforms, validation_transforms)
    return _datasets, _transforms

"""
    Below was referenced from:
    [3] https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/benchmarks/__init__.py
    Goal: Instantiate the tasksets
"""
from collections import namedtuple

BenchmarkTasksets = namedtuple('BenchmarkTasksets', ('train', 'validation'))

def get_cure_tsr_multilevel_inter_tasksets(
    #name,
    inter_dir_list,
    train_ways=5,
    train_samples=10,
    test_ways=5,
    test_samples=10,
    num_tasks=-1,
    
    #root='~/data',
    device=None,
    **kwargs,
):
    #root = os.path.expanduser(root)

    if device is not None:
        raise NotImplementedError('Device other than None not implemented. (yet)')

    # Load task-specific data and transforms
    datasets, transforms = cure_tsr_multilevel_intermediate_tasksets(inter_dir_list, train_ways=train_ways,
                                           train_samples=train_samples,
                                           test_ways=test_ways,
                                           test_samples=test_samples,
                                           **kwargs)
    train_dataset, validation_dataset = datasets
    train_transforms, validation_transforms = transforms

    # Instantiate the tasksets
    train_tasks = l2l.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_tasks,
    )
    validation_tasks = l2l.data.TaskDataset(
        dataset=validation_dataset,
        task_transforms=validation_transforms,
        num_tasks=num_tasks,
    )
    return BenchmarkTasksets(train_tasks, validation_tasks)

#get_cure_tsr_tasksets()
