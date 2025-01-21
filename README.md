# ct2mri2025
Paper's name: Scalable Diffusion Models for Inter-modality Volumetric Medical Image Translation.
Journal: The Visual Computer.

# Abstract 
Recent advancements in generative models, particularly diffusion models, have demonstrated remarkable performance in various image generation tasks. However, translating Computed Tomography (CT) volumes into Magnetic Resonance Imaging (MRI) volumes remains a challenging problem due to the differences in image contrast, resolution, and the high computational demands of three-dimensional data. In this work, we propose a novel Multi-Dimensional Diffusion Generation Architecture (MD-DGA) to address this challenge. MD-DGA integrates a two-dimensional scalable diffusion model (2D-SDM) and a three-dimensional scalable latent diffusion model (3D-SLDM). The 2D-SDM generates detailed MRI slices from paired CT slices, while the 3D-SLDM further refines these slices into complete MRI volumes. To handle diverse input shapes, we introduce a scalable module that adaptively pads inputs at each layer of the model. Experimental results on brain and pelvis datasets demonstrate that our approach outperforms state-of-the-art methods in terms of Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and other metrics. 

# 
