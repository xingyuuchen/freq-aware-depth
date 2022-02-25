import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


class AutoBlurModule(nn.Module):
    def __init__(self, receptive_field_of_hf_area,
                 hf_pixel_thresh=0.2,
                 hf_area_percent_thresh=60,
                 gaussian_blur_kernel_size=11,
                 gaussian_blur_sigma=5.0,
                 ):
        super(AutoBlurModule, self).__init__()

        self.receptive_field_of_hf_area = receptive_field_of_hf_area
        self.hf_pixel_thresh = hf_pixel_thresh
        self.hf_area_ratio = hf_area_percent_thresh / 100

        self.gaussian_blur = transforms.GaussianBlur(gaussian_blur_kernel_size,
                                                     sigma=gaussian_blur_sigma)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=receptive_field_of_hf_area, stride=1,
            padding=(receptive_field_of_hf_area - 1) // 2)

    @staticmethod
    def compute_spatial_grad(ipt):
        grad_u = torch.abs(ipt[:, :, :, :-1] - ipt[:, :, :, 1:]).sum(1, True)
        grad_v = torch.abs(ipt[:, :, :-1, :] - ipt[:, :, 1:, :]).sum(1, True)

        grad_u = F.pad(grad_u, (0, 1))
        grad_v = F.pad(grad_v, (0, 0, 0, 1))

        grad_l2_norm = torch.sqrt(grad_u ** 2 + grad_v ** 2)
        return grad_l2_norm

    def forward(self, raw_img):
        # Gaussian blur the whole image first.
        blurred_img = self.gaussian_blur(raw_img)

        # Whether it is a high frequency pixel.
        spatial_grad = self.compute_spatial_grad(raw_img)
        is_hf_pixel = spatial_grad > self.hf_pixel_thresh

        # Compute how many high frequency pixels are around.
        avg_pool_freq = self.avg_pool(is_hf_pixel.float())

        # If 60% of the surrounding pixels are high frequency,
        # the pixel considered to be in the high frequency region.
        is_in_hf_area = avg_pool_freq > self.hf_area_ratio

        weight_blur = avg_pool_freq * is_in_hf_area

        # Only pixels located in high frequency regions are
        # gaussian blurred, with other pixels unchanged.
        # The more the avg freq, the more the pixel is blurred.
        auto_blurred = blurred_img * weight_blur + \
                       raw_img * (1 - weight_blur)

        return auto_blurred
