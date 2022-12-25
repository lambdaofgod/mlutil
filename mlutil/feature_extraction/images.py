import numpy as np
from torchvision import transforms, models
import torch
from torch import nn
import tqdm
from PIL import Image
from typing import Union, List
from torch.utils import data

imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class TorchFeatureExtractor:
    def __init__(
        self,
        model,
        appended_modules: List[nn.Module] = [nn.Flatten()],
        last_layer_index: int = None,
        last_nested_layer_index: int = None,
        use_gpu=True,
        normalize=imagenet_normalize,
        to_fp16=False,
        img_size=224,
    ):
        self.use_gpu = use_gpu
        self.normalize = normalize
        self.to_fp16 = to_fp16
        self.model = self.get_layers(
            model,
            appended_modules=appended_modules,
            last_layer_index=last_layer_index,
            last_nested_layer_index=last_nested_layer_index,
        )
        self.scaler = transforms.Scale((img_size, img_size))
        self.torch_transforms = [transforms.ToTensor(), self.scaler, self.normalize]

    def torch_image_transform(self, img: Image.Image):
        return self.normalize(transforms.ToTensor()(self.scaler(img)))

    def load_img_tensor(self, path):
        img = Image.fromarray(path)
        return self.process_img(img)

    def process_img(self, img: Union[np.ndarray, Image.Image]):
        if type(img) is np.ndarray:
            if img.dtype is not np.dtype("uint8"):
                img = (img * 255).astype("uint8")
            img = Image.fromarray(img)
        torch_img = self.torch_image_transform(img)
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

    def get_features_from_dataset(
        self, images_dset: data.Dataset, batch_size=32, use_tqdm=True
    ):
        tqdm_wrapper = tqdm.tqdm if use_tqdm else lambda x: x
        data_loader = data.DataLoader(images_dset, batch_size=batch_size)
        features = []
        for batch in tqdm_wrapper(data_loader):
            img_features = self.model(batch)
            features.append(img_features.cpu().numpy())
        return features

    def get_images(self, paths):
        return [self.load_img(p) for p in tqdm.tqdm(paths)]

    def get_layers(
        self, model, appended_modules, last_layer_index, last_nested_layer_index
    ):
        model = model.eval()
        modules = list(model.children())[:last_layer_index]
        if last_nested_layer_index is not None:
            last_module = list(model.children())[-1]
            modules.append(last_module[:last_nested_layer_index])
        model = nn.Sequential(*(modules + appended_modules))
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return self.maybe_to_cuda(model)
