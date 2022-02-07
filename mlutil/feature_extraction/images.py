import numpy as np
from torchvision import transforms, models
import torch
from torch import nn
import tqdm
from PIL import Image
from typing import Union, List


imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class TorchFeatureExtractor:
    def __init__(
        self,
        model,
        last_nested_index=None,
        last_layers=[nn.Flatten()],
        use_gpu=True,
        normalize=imagenet_normalize,
        to_fp16=False,
        img_size=224,
    ):
        self.use_gpu = use_gpu
        self.normalize = normalize
        self.to_fp16 = to_fp16
        self.model = self.get_layers(model, last_nested_index, last_layers)
        self.scaler = transforms.Scale((img_size, img_size))


    def load_img_tensor(self, path):
        img = Image.fromarray(path)
        return self.process_img(img)

    def process_img(self, img: Union[np.ndarray, Image.Image]):
        if type(img) is np.ndarray:
            img = Image.fromarray(img)
        torch_img = self.normalize(transforms.ToTensor()(self.scaler(img)))
        return self.maybe_to_cuda(torch_img.unsqueeze(0))

    def maybe_to_cuda(self, torch_object):
        if self.use_gpu:
            torch_object = torch_object.cuda()
        if self.to_fp16:
            torch_object = torch_object.half()
        return torch_object

    def get_features(self, images: List[Union[Image.Image, np.ndarray]], use_tqdm=True):
        tqdm_wrapper = tqdm.tqdm if use_tqdm else lambda x: x
        features = []
        for img in tqdm_wrapper(images):
            img = self.process_img(img)
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
