from torch.utils import data
from data.FFHQ_dataset import FFHQ
import torchvision.transforms as transforms

def get_train_loader(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = FFHQ(args.train_json_path, args.img_dir, img_transform)

    train_dataloader = data.DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return train_dataloader

def get_val_loader(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = FFHQ(args.val_json_path, args.img_dir, img_transform)

    val_dataloader = data.DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return val_dataloader

def get_test_loader(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = FFHQ(args.test_json_path, args.img_dir, img_transform)

    test_dataloader = data.DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return test_dataloader