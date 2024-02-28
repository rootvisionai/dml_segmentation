from torchvision import transforms
import random
from PIL import ImageFilter
import torch
    
class Resizer(object):

    def __init__(self, input_size, inference = False):
        self.output_smallest_size = input_size
        self.inference = inference

    def __call__(self, sample):
        sample = transforms.functional.resize(img=sample,
                                              size=self.output_smallest_size)
        return sample

class CenterCrop(object):

    def __init__(self, input_size, inference = False):
        
        self.size = input_size
        self.inference = inference
        
    def __call__(self, sample):
        sample = transforms.functional.center_crop(img=sample,
                                                   output_size=self.size)
        return sample

class PadToSquare(object):

    def __init__(self, inference = False):
        self.inference = inference

        self.width_pad = False
        self.height_pad = False

    def __call__(self, sample):
        input_width, input_height = sample.size
        if input_width<input_height:
            self.width_pad = int((input_height - input_width)/2)
            self.height_pad = 0
        elif input_height<input_width:
            self.height_pad = int((input_width - input_height)/2)
            self.width_pad = 0
        else:
            self.height_pad = 0
            self.width_pad = 0

        sample = transforms.functional.pad(img = sample,
                                                    padding = (self.width_pad,
                                                               self.height_pad),
                                                    fill = 0,
                                                    padding_mode = 'constant')
        return sample

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        
        if torch.rand(1)[0] < self.p:
            sigma = random.random() * 1.9 + 0.1
            x = x.filter(ImageFilter.GaussianBlur(sigma))
        else:
            pass
        
        return x

class RandomVerticalFlip(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):

        if torch.rand(1)[0] < self.p:
            sample = transforms.functional.vflip(sample)
        return sample
    
class RandomHorizontalFlip(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):

        if torch.rand(1)[0] < self.p:
            sample = transforms.functional.hflip(sample)
        return sample

class ColorJitter(object):
    def __init__(self,
                 brightness=0.4,
                 contrast=0.4,
                 saturation=0.01,
                 hue=0.01, p=0.2):

        self.p = p
        self.apply = transforms.ColorJitter(brightness=brightness,
                                            contrast=contrast,
                                            saturation=saturation,
                                            hue=hue)

    def __call__(self, x):
        
        if random.random() < self.p:
            x = self.apply(x)
        else:
            pass
        
        return x


class RandomResizedCrop(object):
    def __init__(self, input_size, p, scale=(0.5, 1), ratio=(0.75, 1.33)):

        self.p = p
        self.apply = transforms.RandomResizedCrop(size=input_size,
                                                  scale=scale,
                                                  ratio=ratio)

    def __call__(self, x):

        if random.random() < self.p:
            x = self.apply(x)
        else:
            pass

        return x

class RandomGrayscale(object):
    def __init__(self,
                 p=0.2):
        
        self.p = p
        self.apply = transforms.RandomGrayscale()

    def __call__(self, x):
        
        if random.random() < self.p:
            x = self.apply(x)
        else:
            pass
        
        return x

class ToTensor(object):
    def __init__(self):
        self.apply = transforms.ToTensor()
    
    def __call__(self, x):
        x = self.apply(x)
        return x
    
class Normalize(object):
    def __init__(self, mean, std):
        self.apply = transforms.Normalize(mean, std)
    
    def __call__(self, x):
        x = self.apply(x)
        return x

class Identity(object):
    def __call__(self, x):
        return x

class TransformTrain:
    
    def __init__(self, cfg):
        self.transform = transforms.Compose([
            PadToSquare(inference=False) if cfg.data.preprocessing.pad_to_square else Identity(),

            RandomResizedCrop(
                input_size=cfg.data.preprocessing.input_size,
                p=cfg.data.preprocessing.random_resized_crop.p,
                scale=cfg.data.preprocessing.random_resized_crop.scale,
                ratio=cfg.data.preprocessing.random_resized_crop.ratio
            ),

            Resizer(cfg.data.preprocessing.input_size, inference=False),

            CenterCrop(cfg.data.preprocessing.input_size, inference=False),

            ColorJitter(
                p=cfg.data.preprocessing.color_jitter.p,
                brightness=cfg.data.preprocessing.color_jitter.brightness,
                contrast=cfg.data.preprocessing.color_jitter.contrast,
                saturation=cfg.data.preprocessing.color_jitter.saturation,
                hue=cfg.data.preprocessing.color_jitter.hue
            ) if cfg.data.preprocessing.color_jitter.p else Identity(),

            RandomGrayscale(
                p=cfg.data.preprocessing.random_gray_scale.p,
            ) if cfg.data.preprocessing.random_gray_scale.p else Identity(),

            GaussianBlur(
                p=cfg.data.preprocessing.gaussian_blur.p
            ) if cfg.data.preprocessing.gaussian_blur.p else Identity(),

            ToTensor(),

            RandomVerticalFlip(p=cfg.data.preprocessing.random_vertical_flip.p) \
                if cfg.data.preprocessing.random_vertical_flip.p else Identity(),
            RandomHorizontalFlip(p=cfg.data.preprocessing.random_horizontal_flip.p) \
                if cfg.data.preprocessing.random_horizontal_flip.p else Identity(),

            Normalize(
                mean=cfg.data.preprocessing.normalize.mean,
                std=cfg.data.preprocessing.normalize.std
            ) if cfg.data.preprocessing.normalize.p else Identity()
        ])

    def __call__(self, x):
        y = self.transform(x)
        return y
    
class TransformEvaluate:
    def __init__(self, cfg):
        self.transform = transforms.Compose([
            PadToSquare(inference=False) if cfg.data.preprocessing.pad_to_square else Identity(),
            Resizer(cfg.data.preprocessing.input_size, inference=False),
            CenterCrop(cfg.data.preprocessing.input_size, inference=False),
            ToTensor(),
            Normalize(
                mean=cfg.data.preprocessing.normalize.mean,
                std=cfg.data.preprocessing.normalize.std
            ) if cfg.data.preprocessing.normalize.p else Identity()
        ])
    def __call__(self, x):
        y = self.transform(x)
        return y
    
class TransformInference:
    def __init__(self, cfg):
        self.transform = transforms.Compose([
            PadToSquare(inference=True) if cfg.data.preprocessing.pad_to_square else Identity(),
            Resizer(cfg.data.preprocessing.input_size, inference=True),
            CenterCrop(cfg.data.preprocessing.input_size, inference=True),
            ToTensor(),
            Normalize(
                mean=cfg.data.preprocessing.normalize.mean,
                std=cfg.data.preprocessing.normalize.std
            ) if cfg.data.preprocessing.normalize.p else Identity()
        ])
    def __call__(self, x):
        y = self.transform(x)
        return y
