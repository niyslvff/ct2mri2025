import os
import sys
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append("..")

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional
import argparse
import datetime


class Train:
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.dims = kwargs.get("dims")
        self.crop = kwargs.get("crop")
        self.epoch_num = kwargs.get("epoch_num")
        self.batch_size = kwargs.get("batch_size")        
        self.learning_rate = kwargs.get("learning_rate")
        self.model_name = kwargs.get("model_name")
        self.in_channels = kwargs.get("in_channels")
        self.embedding_dim = kwargs.get("embedding_dim")
        self.latent_dim = kwargs.get("latent_dim")
        self.image_size = kwargs.get("image_size")
        self.image_root = kwargs.get("image_root")
        self.organ = kwargs.get("organ")
        self.create_time = kwargs.get("create_time")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.average_loss_log = {"val": 0, "avg": 0, "sum": 0, "count": 0}
        self.current_epoch = 0
        self.best_epoch = 0
        self.current_loss = 99999
        self.best_loss = 99999
        

        self.dataset = self.choose_dataset(image_root=self.image_root, image_size=self.image_size)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
            drop_last=True,
        )
        
        self.model = self.choose_model()

        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        
    def choose_dataset(self, image_root: str, image_size: tuple):
        if self.dims == 2:
            from dataset import MixedSize2DDataset
            return MixedSize2DDataset(
                file_path=image_root, # "../MIXED_2D_IMGS(MIXED_SIZE).hdf5"
                crop=self.crop,
                is_val=False,
                my_transform=None,
            )
        elif self.dims == 3:
            from dataset import Raw3dMixedDiffusionDataset
            return Raw3dMixedDiffusionDataset(
                img_root_path=self.image_root, # '../crop/preprocess_globalNormAndEnContrast'
                img_shape=self.image_size,
                use_crop=self.crop,
                img_type="mixed",
                transform=None,
            )
        else:
            ValueError("wrong dims.")
        
    def choose_model(self):
        if self.model_name == "default":
            from models.vae.vae import VanillaVAE
            return VanillaVAE(
                in_channels=1,
                latent_dim=self.latent_dim
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
            
            

    def reset_average_loss_log(self):
        r"""重置loss记录。
        """
        for key, _ in self.average_loss_log.items():
            self.average_loss_log[key] = 0

    def update_average_loss_log(self, val, n=1):
        r"""更新loss记录。
        """
        self.average_loss_log["val"] = val
        self.average_loss_log["sum"] += val * n
        self.average_loss_log["count"] += n
        self.average_loss_log["avg"] = (
            self.average_loss_log["sum"] / self.average_loss_log["count"]
        )

    def dataloader_collate_fn_2d(self, batch):
        pass
    

    def __call__(self):
        print("start training.")
        if self.create_time is None:
            self.create_time = datetime.datetime.now().strftime("%Y%m%d")
        self.checkpoints_root_path = f"./{self.model_name}/weight_file/{self.organ}/{self.model_name}.{self.dims}.{self.create_time}"
        if not os.path.exists(self.checkpoints_root_path):
            os.makedirs(self.checkpoints_root_path)
        
        self.model.to(self.device)
        self.load_checkpoints(model_name=self.model_name)
        self.model.train()
        

        for epoch in range(self.current_epoch, self.epoch_num):
            with tqdm(total=self.dataset.__len__() - self.dataset.__len__() % self.batch_size) as tq:
                tq.set_description(f"epoch: {epoch}/{self.epoch_num} ")
                self.reset_average_loss_log()
                for batch in self.dataloader:
                    img, _ = batch
                    img = img.to(self.device)
                    
                    self.optimizer.zero_grad()
                    out_list = self.model(img)
                    if self.model_name == "default":
                        decoded_input, _input, mu, log_var = out_list[0], out_list[1], out_list[2], out_list[3]
                        loss_dict = self.model.loss_function(decoded_input, _input, mu, log_var, M_N=0.00025)
                    elif self.model_name == "vqvae":
                        recons, _input, vq_loss = out_list[0], out_list[1], out_list[2]
                        loss_dict = self.model.loss_function(recons, _input, vq_loss)
                    loss = loss_dict['loss']
                    self.update_average_loss_log(loss.item(), self.batch_size)
                    loss.backward()
                    self.optimizer.step()

                    tq.set_postfix(
                        loss="{:.6f}".format(self.average_loss_log["avg"]),
                        best_epoch=self.best_epoch,
                        best_loss=self.best_loss,
                    )
                    tq.update(self.batch_size)
            # break
            self.save_checkpoints(
                current_epoch=epoch, current_loss=self.average_loss_log["avg"], model_name=self.model_name
            )
        print("best epoch: {}, loss: {:.5f}".format(self.best_epoch, self.best_loss))

    def load_checkpoints(self, model_name: str):
        r"""加载权重。
        """
        checkpoints_root_path = self.checkpoints_root_path
        checkpoints_path = checkpoints_root_path + "/latest.pth"
        if os.path.exists(checkpoints_path):
            checkpoints_file = torch.load(checkpoints_path)
            self.best_epoch = checkpoints_file["best_epoch"]
            self.current_epoch = checkpoints_file["current_epoch"]
            self.best_loss = checkpoints_file["best_loss"]
            self.current_loss = checkpoints_file["current_loss"]
            self.model.load_state_dict(checkpoints_file["model_parameters"])
            base_learning_rate = self.optimizer.param_groups[0]["lr"]
            self.optimizer.load_state_dict(checkpoints_file["optimizer_parameters"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = base_learning_rate

            print(f"Start from epoch {self.current_epoch}, loss {self.current_loss}.")
            print(
                f"Best epoch {self.best_epoch}, best loss {self.best_loss}.\n", 20 * "="
            )
        else:
            print("Start from zero.")

    def save_checkpoints(self, current_loss: float, current_epoch: int, model_name: str):
        r"""保存权重。
        """
        checkpoints_latest_path = self.checkpoints_root_path + "/latest.pth"
        checkpoints_best_path = self.checkpoints_root_path + "/best.pth"
        save_best_model = False

        self.current_epoch = current_epoch
        self.current_loss = current_loss
        if self.best_loss >= self.current_loss:
            self.best_loss = self.current_loss
            self.best_epoch = self.current_epoch
            save_best_model = True
        state = {
            "model_parameters": self.model.state_dict(),
            "optimizer_parameters": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "current_loss": self.current_loss,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
        }
        torch.save(state, checkpoints_latest_path)
        if save_best_model:
            torch.save(state, checkpoints_best_path)

    @staticmethod
    def show_image(current_tensor):
        current_image = current_tensor[0][0].detach().cpu()
        print(current_image.shape)
        plt.figure()
        plt.imshow(current_image[80, :, :])
        plt.savefig("train_vae_result.png")
        plt.close()


class BaseCommand:
    r"""训练vae
    训练头颅的vqvae: python3 train_vae.py --crop=1 --embedding_dim=64
    训练骨盆的vqvae: python3 train_vae.py --depth=96 --height=240 --width=384 --image_root=../crop/preprocess_globalNormAndEnContrast_pelvis --crop=1
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="基础命令行解析器")
        self.add_base_arguments()

    def add_base_arguments(self):
        # 添加基础参数
        self.parser.add_argument('--organ', type=str, choices=["brain", "pelvis"], default="brain")
        self.parser.add_argument('--depth', type=int, default=160)
        self.parser.add_argument('--height', type=int, default=224)
        self.parser.add_argument('--width', type=int, default=168)
        self.parser.add_argument('--image_root', type=str, default="../crop/preprocess_globalNormAndEnContrast")
        self.parser.add_argument('--model_name', type=str, default='vqvae')
        self.parser.add_argument('--in_channels', type=int, default=1)
        self.parser.add_argument('--embedding_dim', type=int, default=2)
        self.parser.add_argument('--latent_dim', type=int, default=128)
        self.parser.add_argument('--learning_rate', type=float, default=0.00002)
        self.parser.add_argument('--dims', type=int, default=3)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--epoch_num', type=int, default=8000)
        self.parser.add_argument('--crop', type=int, choices=[0, 1], default=0)
        self.parser.add_argument('--create_time', type=str, default=None)
        
    def parse_args(self):
        args = self.parser.parse_args()
        args.crop = True if int(args.crop) else False
        args.image_size = (args.depth, args.height, args.width)
        return args
    

if __name__ == "__main__":
    base_parser = BaseCommand()
    args = base_parser.parse_args()
    print(args)
    
    train = Train(**vars(args))
    train()

