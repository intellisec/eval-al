import torch
import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser

# Allowed image suffixes
IMG_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize(40),
    transforms.CenterCrop(32)
])

parser = ArgumentParser(description = 'Rescale and crop images to 32x32 px')
parser.add_argument('root',   type = str, help = 'root folder where subdirectories and images are stored')
parser.add_argument('target', type = str, nargs = '?', help = 'target folder name to where preprocessed images should be stored')

def preprocess_image(path, target_path) :
    with path.open('rb') as img :
        # Load and transform image
        img = Image.open(img)
        img = img.convert('RGB')
        img = transform(img)

        # Save image in target directory
        img.save(target_path)

        # Calculate mean and std
        data = transforms.ToTensor()(img)
        mean = data.mean(dim = (1, 2))
        std  = data.std(dim = (1, 2))
    return mean, std

def traverse(path, root, target) :
    # Replicate directory structure
    target_path = target / path.relative_to(root)
    target_path.mkdir(exist_ok = True)
    
    # Accumulate means and standard deviations
    means = torch.tensor([])
    stds  = torch.tensor([])

    print(f'Processing {path}...')
    for subpath in path.iterdir() :
        if subpath.is_dir() :
            # Descent deeper into the directory tree
            mean, std = traverse(subpath, root, target)
        elif subpath.suffix.lower() in IMG_SUFFIXES :
            # Preprocess and save image
            target_path = target / subpath.relative_to(root)
            try :
                mean, std = preprocess_image(subpath, target_path)
                mean, std = mean.unsqueeze(dim = 0), std.unsqueeze(dim = 0)
            except Exception as e :
                print(e)
                continue
        else :
            continue
        # import pdb; pdb.set_trace()
        means = torch.cat([means, mean])
        stds  = torch.cat([stds, std])
    return means, stds
            
def main(root, target = None) :
    # Prepare paths
    root = Path(root)
    if not target :
        target = root.name + '_pp'
    target = Path(target)

    # Traverse folder and preprocess images
    means, stds = traverse(root, root, target)
    mean = means.mean(dim = 0)
    std = stds.mean(dim = 0)
    print(f'Images mean: {mean}')
    print(f'Images std:  {std}')

if __name__ == '__main__' :
    args = parser.parse_args()
    main(**vars(args))