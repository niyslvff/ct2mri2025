import torch
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
import datetime
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append("..")

class Test:
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.in_channels = kwargs.get("in_channels")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dims = kwargs.get("dims")
        self.model_name = kwargs.get("model_name")
        self.version = kwargs.get("version")
        self.image_size = kwargs.get("image_size")
        self.image_root = kwargs.get("image_root")
        self.crop = kwargs.get("crop")
        self.latent_dim = kwargs.get("latent_dim")
        self.embedding_dim = kwargs.get("embedding_dim")
        self.organ = kwargs.get("organ")
        
        self.model = self.choose_model().to(self.device)
        weight_file_path = f"{kwargs.get('weight_file_root')}/{self.model_name}/weight_file/{self.organ}/{self.version}/best.pth"
        checkpoints = torch.load(weight_file_path)
        self.model.load_state_dict(checkpoints["model_parameters"])
        print(f"model best epoch: {checkpoints['best_epoch']}. best loss: {checkpoints['best_loss']}")
        self.model.eval()
        
        self.dataset = self.choose_dataset()
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        
    def choose_model(self):
        if self.model_name == "vae":
            from models.vae.vae import VanillaVAE
            return VanillaVAE(
                in_channels=1,
                latent_dim=self.latent_dim,
            )
        elif self.model_name == "vqvae":
            from models.vqvae.vq_vae import VQVAE
            return VQVAE(
                in_channels=1,
                out_channels=1,
                embedding_dim=self.embedding_dim,
                num_embeddings=512,
                beta=0.25,
                dims=self.dims,
                change_channels=False,
            )
        else:
            ValueError("wrong model name.")
        
    def choose_dataset(self):
        if self.dims == 2:
            from dataset import MixedSize2DDataset
            return MixedSize2DDataset(
                file_path="../MIXED_2D_IMGS(MIXED_SIZE).hdf5",
                crop=self.crop,
                is_val=False,
                my_transform=None,
            )
        elif self.dims == 3:
            from dataset import Raw3dDiffusion160224168Dataset
            return Raw3dDiffusion160224168Dataset(
                img_root_path=self.image_root,
                img_shape=self.image_size,
                use_crop=self.crop,
                is_val=True,
                transform=None,
            )
        else:
            ValueError("wrong dims.")
        

    @torch.no_grad()
    def test_loop(self):
        # 创建预测结果文件夹
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ct_save_path = f"prediction/{self.organ}/{self.model_name}_{self.version}_test.{current_time}/ct"
        mri_save_path = f"prediction/{self.organ}/{self.model_name}_{self.version}_test.{current_time}/mri"  
        os.makedirs(ct_save_path)
        os.makedirs(mri_save_path)
        
        with tqdm(total=len(self.dataset)) as tq:
            for idx, batch in enumerate(self.dataloader):
                ct_volume, mri_volume = batch
                ct_image, mri_image = ct_volume.squeeze().squeeze(), mri_volume.squeeze().squeeze()
                ct_volume, mri_volume = ct_volume.to(self.device), mri_volume.to(self.device)

                decoded_ct_volume = self.model.generate(ct_volume).cpu()
                decoded_ct_volume = decoded_ct_volume.squeeze().squeeze()
                decoded_mri_volume = self.model.generate(mri_volume).cpu()
                decoded_mri_volume = decoded_mri_volume.squeeze().squeeze()
                
                decoded_ct_volume.numpy().tofile(f"{ct_save_path}/{idx}.raw")
                decoded_mri_volume.numpy().tofile(f"{mri_save_path}/{idx}.raw")

                tq.update(1)

class BaseCommand:
    r"""
    测试头颅的vqvae_v1: python3 test_vae.py --model_name=vqvae --dims=3 --crop=0 --version=v1
    测试骨盆的vqvae_v1: python3 test_vae.py --organ=pelvis --model_name=vqvae --dims=3 --crop=0 --version=v1 --depth=96 --height=240 --width=384 --image_root=../crop/preprocess_globalNormAndEnContrast_pelvis
    
    """
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        
    def add_base_arguments(self):
        self.parser.add_argument('--weight_file_root', type=str, default="../train")
        self.parser.add_argument('--organ', type=str, choices=["brain", "pelvis"], default="brain")
        self.parser.add_argument('--depth', type=int, default=160)
        self.parser.add_argument('--height', type=int, default=224)
        self.parser.add_argument('--width', type=int, default=168)
        self.parser.add_argument('--image_root', type=str, default="../crop/preprocess_globalNormAndEnContrast")
        self.parser.add_argument('--model_name', type=str, default="vae")
        self.parser.add_argument('--version', type=str, default="v1")
        self.parser.add_argument('--embedding_dim', type=int, default=2)
        self.parser.add_argument('--latent_dim', type=int, default=128)
        self.parser.add_argument('--in_channels', type=int, default=1)
        self.parser.add_argument('--dims', type=int, choices=[2, 3], default=3)
        self.parser.add_argument('--crop', type=int, choices=[0, 1], default=0)
        
        
    def parse_args(self):
        args = self.parser.parse_args()
        args.crop = bool(int(args.crop))
        args.image_size = (args.depth, args.height, args.width)
        return args
        


if __name__ == "__main__":
    command_parse = BaseCommand()
    args = command_parse.parse_args()
    print(args)
    
    print("是否开始测试？(1/0)")
    if int(input()) == 1:
        t = Test(**vars(args))
        t.test_loop()
