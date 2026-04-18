'''
transfrom the image
'''
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

import numpy as np
from datautil.util import Nmax
from datautil.imgdata.util import rgb_loader, l_loader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random
    

class ImageDataset(object):
    def __init__(self, dataset, root_dir, domain_name, domain_label= -1, labels= None, transform=None, 
                    target_transform= None, indices= None, test_envs= [], mode= 'Default', CO=False) -> None:
        
        img_folder = ImageFolder(root_dir + domain_name)
        
        self.imgs = img_folder.imgs
        self.domain_num = 0
        self.dataset = dataset
        imgs = [item[0] for item in self.imgs]
        labels = [item[1] for item in self.imgs]
        self.labels = np.array(labels)
        ######
        self.domain_bank = [
            "a picture of a "
            "an image of a "
            "a photograph of a ",
            "a painting of a ",
            "a sketch of a ",
            "a cartoon of a ",
            "a 3D render of a ",
            "a drawing of a ",
            "a grayscale image of a ",
            "a low-light image of a ",
            "a high-resolution image of a ",
            "a blurred image of a ",
            "an overexposed image of a ",
            "a noisy image of a ",
            "a close-up image of a ",
            "a wide-angle image of a ",
            "an indoor image of a ",
            "an outdoor image of a "
        ]
        # self.domain_bank = [
        #     "an image of a "
        # ]

        self.text_labels = [img_folder.classes[item[1]] for item in self.imgs]
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

        unique_classes = sorted(list(set(self.text_labels)))
        num_classes = len(unique_classes)
        num_domains = len(self.domain_bank)

        all_prompts_nested = [
            [f"{prefix}{class_name}" for prefix in self.domain_bank]
            for class_name in unique_classes
        ]

        flat_prompts = [prompt for sublist in all_prompts_nested for prompt in sublist]
        tokenized_output = self.clip_processor(
            text=flat_prompts,
            return_tensors="pt",   
            padding='max_length',  
            truncation=True
        )
        sequence_length = tokenized_output.input_ids.shape[1]

        self.input_ids = tokenized_output.input_ids.view(
            num_classes,
            num_domains,
            sequence_length  # or -1 to infer automatically
        )

        self.attention_mask = tokenized_output.attention_mask.view(
            num_classes,
            num_domains,
            sequence_length
        )

        ######
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.CO = CO
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        if mode == 'Default':
            self.loader = default_loader
        elif mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.dlabels = np.ones(self.labels.shape) * \
            (domain_label - Nmax(test_envs, domain_label))
        
    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y
    
    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)

        else: 
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        img_q = self.input_trans(self.loader(self.x[index]))   # the quere image in CO
        text_label = self.text_labels[index]
        ctarget = self.target_trans(self.labels[index])
        dtarget = self.target_trans(self.dlabels[index])

        random_domain_idx = random.choice(range(0, len(self.domain_bank)))
        input_ids = self.input_ids[self.labels[index], random_domain_idx, :]
        attention_mask = self.attention_mask[self.labels[index], random_domain_idx, :]
        tokenized_prompt = {'input_ids': input_ids, 'attention_mask': attention_mask}

        ####
        if self.CO:
            # print("CO mode")
            img_k = self.input_trans(self.loader(self.x[index]))    # the key image in CO
            return img_q, tokenized_prompt, ctarget, dtarget, img_k
        else:
            # print("not CO mode")
            return img_q, tokenized_prompt, ctarget, dtarget

    def __len__(self):
        return len(self.indices)
