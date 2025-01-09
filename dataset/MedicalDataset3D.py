from typing import Callable
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision.transforms import Compose
import torchvision

class MedicalDataset3D(Dataset):
    r"""modified on 2024.12.30."""
    
    def __init__(
        self,
        image_root: str = "../crop/preprocess_globalNormAndEnContrast",
        image_size: tuple[int, int, int] = (160, 224, 168),
        validate: bool = False,
        transform: Callable = None,
    ) -> None:
        super().__init__()
        ct_image_names = os.listdir(image_root + "/ct")
        ct_image_names.sort(key=lambda x: int(x.split('.')[1]))
        mr_image_names = os.listdir(image_root + "/mr")
        mr_image_names.sort(key=lambda x: int(x.split('.')[1]))

        assert len(ct_image_names) == len(
            mr_image_names
        ), "image amounts are not matched."
        self.image_total_amount = len(ct_image_names)

        ct_image_paths = [os.path.join(image_root, "ct", nm) for nm in ct_image_names]
        mr_image_paths = [os.path.join(image_root, "mr", nm) for nm in mr_image_names]

        training_rate = 0.7

        training_index_end = int(self.image_total_amount * training_rate)
        using_training_images_amount = training_index_end
        using_validating_images_amount = (
            self.image_total_amount - using_training_images_amount
        )
        self.using_images_amount = (
            using_training_images_amount
            if not validate
            else using_validating_images_amount
        )

        training_ct_images = ct_image_paths[:training_index_end]
        training_mr_images = mr_image_paths[:training_index_end]
        validating_ct_images = ct_image_paths[training_index_end:]
        validating_mr_images = mr_image_paths[training_index_end:]
        self.using_ct_images = (
            training_ct_images if not validate else validating_ct_images
        )
        self.using_mr_images = (
            training_mr_images if not validate else validating_mr_images
        )

        if transform:
            self.transform = transform
        else:
            self.transform = Compose([
                torchvision.transforms.ToTensor(),
            ])
        self.image_size = image_size

    def __getitem__(self, index):
        # (160, 224, 168) -> (168, 160, 224)
        ct_tensor = self.transform(np.reshape(np.fromfile(self.using_ct_images[index % self.using_images_amount],dtype=np.float32,),self.image_size,))
        mr_tensor = self.transform(np.reshape(np.fromfile(self.using_mr_images[index % self.using_images_amount],dtype=np.float32,),self.image_size,))
        # (168, 160, 224) -> (160, 224, 168)
        ct_tensor = ct_tensor.permute(1, 2, 0)
        mr_tensor = mr_tensor.permute(1, 2, 0)
        return ct_tensor, mr_tensor

    def __len__(self):
        return self.using_images_amount

if __name__ == "__main__":
    mydataset = MedicalDataset3D()
    test_batch = next(iter(mydataset))
    ct, mr = test_batch
    print(test_batch[0].shape)

    import matplotlib.pyplot as plt
    plt.subplot(231)
    plt.imshow(ct[:, ct.shape[1] // 2, :, :][0])
    plt.subplot(232)
    plt.imshow(ct[:, :, ct.shape[2] // 2, :][0])
    plt.subplot(233)
    plt.imshow(ct[:, :, :, ct.shape[3] // 2][0])
    plt.subplot(234)
    plt.imshow(mr[:, mr.shape[1] // 2, :, :][0])
    plt.subplot(235)
    plt.imshow(mr[:, :, mr.shape[2] // 2, :][0])
    plt.subplot(236)
    plt.imshow(mr[:, :, :, mr.shape[3] // 2][0])
    plt.savefig("MedicalDataset_test.png")

