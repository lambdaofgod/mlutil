import numpy as np
from torchvision import transforms, models
import torch
from torch import nn
import tqdm
from PIL import Image


imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class TorchFeatureExtractor:
    
    def __init__(
            self,
            model,
            last_nested_index=None,
            last_layers=[nn.Flatten()],
            use_gpu=True,
            normalize=imagenet_normalize,
            to_fp16=False,
            img_size=224
        ):
        self.use_gpu = use_gpu
        self.normalize = normalize
        self.to_fp16 = to_fp16
        self.model = self.get_layers(model, last_nested_index, last_layers)
        self.scaler = transforms.Scale((img_size, img_size))
    
    def load_img(self, path):
        img = Image.open(path)
        torch_img = self.normalize(transforms.ToTensor()(self.scaler(img)))
        return torch.autograd.Variable(torch_img.unsqueeze(0))
    
    def get_vector(self, image):
        img = self.maybe_to_cuda(load_img(p))
        return self.model(img).cpu().numpy()
        
    def maybe_to_cuda(self, torch_object):
        if self.use_gpu:
            torch_object = torch_object.cuda()
        if self.to_fp16:
            torch_object = torch_object.half()
        return torch_object
        
    def get_features(self, images, use_tqdm=True):
        tqdm_wrapper = tqdm.tqdm if use_tqdm else lambda x: x
        features = []
        for img in tqdm_wrapper(images):
            img = self.maybe_to_cuda(img)
            img_features = self.model(img)
            features.append(img_features.cpu().numpy())
        return np.vstack(features)
    
    def get_images(self, paths):
        return [self.load_img(p) for p in tqdm.tqdm(paths)]
    
    def get_layers(self, model, last_nested_index, last_layer_modules):
        modules = list(model.children())[:-1]
        if last_nested_index is not None:
            last_module = list(model.children())[-1]
            modules.append(last_module[:last_nested_index])
        model = nn.Sequential(*modules + last_layer_modules)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return self.maybe_to_cuda(model)
