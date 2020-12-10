from torchvision import transforms


def get_transform(affine, normalize, rotate):
    train_transform_list = []
    val_transform_list = []
    if affine:
        train_transform_list.append(transforms.RandomAffine((-10,10)))
    if rotate:
        train_transform_list.append(transforms.RandomRotation(45))
        # train_transform_list.append(transforms.RandomCrop(24))
        train_transform_list.append(transforms.ToTensor())
        # train_transform_list.append(transforms.Resize(28,3))
        # val_transform_list.append(transforms.RandomCrop(24))
        val_transform_list.append(transforms.ToTensor())
        # val_transform_list.append(transforms.Resize(28,3))
    else:
        train_transform_list.append(transforms.ToTensor())
        val_transform_list.append(transforms.ToTensor())
    if normalize:
        train_transform_list.append(transforms.Normalize((0.5),(0.5)))
        val_transform_list.append(transforms.Normalize((0.5), (0.5)))

    return transforms.Compose(train_transform_list), transforms.Compose(val_transform_list)
