import torch

ROUNDING_MODE_UP = "Pad"
ROUNDING_MODE_DOWN = "Crop"

def round_value(value: int, multiple: int, up: bool) -> int:
    if multiple == 1:
        return value
    rounded = value // multiple * multiple
    if up:
        if value % multiple != 0:
            rounded += multiple
        assert(rounded >= value)
    else:
        assert(rounded <= value)
    assert(rounded % multiple == 0)
    assert(abs(rounded - value) < multiple)
    return rounded

def split_value(value, split):
    assert(0.0 <= split and split <= 1.0)
    s0 = round(value * split)
    s1 = value - s0
    assert((s0 <= 0 and s1 <= 0) or (s0 >= 0 and s1 >= 0))
    assert(value == s0 + s1)
    return (s0, s1)

def _round_image(
        images: torch.Tensor, 
        round_type: str, 
        nearest_x: int, 
        nearest_y: int, 
        split_x: float, 
        split_y: float, 
        pad_value: float
):
    if nearest_x == 1 and nearest_y == 1:
        return (images, 0, 0, 0, 0)
    (count, original_y, original_x, colors) = images.size()
    round_up = round_type == ROUNDING_MODE_UP
    rounded_x = round_value(original_x, nearest_x, round_up)
    rounded_y = round_value(original_y, nearest_y, round_up)
    delta_x = rounded_x - original_x
    delta_y = rounded_y - original_y
    if round_type == ROUNDING_MODE_UP:
        assert(delta_x >= 0)
        assert(delta_y >= 0)
    elif round_type == ROUNDING_MODE_DOWN:
        assert(delta_x <= 0)
        assert(delta_y <= 0)
    else:
        raise NotImplementedError("Unknown rounding mode!")
    if delta_x + original_x <= 0:
        raise ValueError("Image X dimension must be non-zero!")
    if delta_y + original_y <= 0:
        raise ValueError("Image Y dimension must be non-zero!")
    
    if delta_x == 0 and delta_y == 0:
        return (images, 0, 0, 0, 0)
    left, right = split_value(delta_x, split_x)
    bottom, top = split_value(delta_y, split_y)
    
    if round_up:
        padding = (0, 0, left, right, top, bottom)
        memory_format = None
        if images.is_contiguous():
            memory_format = torch.contiguous_format
        #else:
        #    raise NotImplementedError("Unknown memory format!")
        new_images = torch.empty(
            size=(count, rounded_y, rounded_x, colors),
            dtype=images.dtype,
            layout=images.layout,
            device=images.device,
            requires_grad=images.requires_grad,
            pin_memory=images.is_pinned(),
            memory_format=memory_format
        )
        for i in range(count):
            new_images[i] = torch.nn.functional.pad(
                images[i], 
                pad=padding, 
                value=pad_value
            )
            return (new_images, left, right, bottom, top)
    else:
        cropped_images = images[
            :,
            abs(bottom):original_y-abs(top),
            abs(left):original_x-abs(right),
            :,
        ]#.clone()
        return (cropped_images, left, right, bottom, top)

class ImageRound:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "round_image"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "images": ("IMAGE",),
                # round
                "round_type": (
                [
                    ROUNDING_MODE_UP,
                    ROUNDING_MODE_DOWN,
                ], {
                    "default": ROUNDING_MODE_UP,
                }),
                "nearest_x": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "nearest_y": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                # pad
                "split_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "split_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    def round_image(
            self, 
            images: torch.Tensor, 
            round_type: str, 
            nearest_x: int, 
            nearest_y: int, 
            split_x: float, 
            split_y: float, 
            pad_value: float
    ):
        results = _round_image(
            images=images,
            round_type=round_type,
            nearest_x=nearest_x,
            nearest_y=nearest_y,
            split_x=split_x,
            split_y=split_y,
            pad_value=pad_value,
        )
        return (results[0],)

class ImageRoundAdvanced:
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "crop_left", "crop_right", "crop_bottom", "crop_top")
    FUNCTION = "round_image"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "images": ("IMAGE",),
                # round
                "round_type": (
                [
                    ROUNDING_MODE_UP,
                    ROUNDING_MODE_DOWN,
                ], {
                    "default": ROUNDING_MODE_UP,
                }),
                "nearest_x": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "nearest_y": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
                # pad
                "split_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "split_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "nearest_x_override": ("INT", {"forceInput": True, "default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "nearest_y_override": ("INT", {"forceInput": True, "default": 1, "min": 1, "max": 0xffffffffffffffff}),
                "split_x_override": ("FLOAT", {"forceInput": True, "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "split_y_override": ("FLOAT", {"forceInput": True, "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_value_override": ("FLOAT", {"forceInput": True, "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    def round_image(
            self, 
            images: torch.Tensor, 
            # default/fallback
            round_type: str, 
            nearest_x: int, 
            nearest_y: int, 
            split_x: float, 
            split_y: float, 
            pad_value: float, 
            # optional input overrides
            nearest_x_override: int = None,
            nearest_y_override: int = None,
            split_x_override: float = None, 
            split_y_override: float = None, 
            pad_value_override: float = None, 
    ):
        return _round_image(
            images=images,
            round_type=round_type,
            nearest_x=(nearest_x_override if type(nearest_x_override) is int else nearest_x),
            nearest_y=(nearest_y_override if type(nearest_y_override) is int else nearest_y),
            split_x=(split_x_override if type(split_x_override) is float else split_x),
            split_y=(split_y_override if type(split_y_override) is float else split_y),
            pad_value=(pad_value_override if type(pad_value_override) is float else pad_value),
        )

class ImageCropAdvanced:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "crop_image"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "images": ("IMAGE",),
                "crop_left": ("INT", {"default": 0}),
                "crop_right": ("INT", {"default": 0}),
                "crop_bottom": ("INT", {"default": 0}),
                "crop_top": ("INT", {"default": 0}),
            },
            "optional": {
                "crop_left_override": ("INT", {"forceInput": True, "default": 0}),
                "crop_right_override": ("INT", {"forceInput": True, "default": 0}),
                "crop_bottom_override": ("INT", {"forceInput": True, "default": 0}),
                "crop_top_override": ("INT", {"forceInput": True, "default": 0}),
            },
        }

    def crop_image(
        self,
        images: torch.Tensor, 
        # default/fallback
        crop_left: int, 
        crop_right: int, 
        crop_bottom: int, 
        crop_top: int, 
        # optional input overrides
        crop_left_override: int = None,
        crop_right_override: int = None,
        crop_bottom_override: int = None,
        crop_top_override: int = None,
    ):
        crop_left = crop_left_override if type(crop_left_override) is int else crop_left
        crop_right = crop_right_override if type(crop_right_override) is int else crop_right
        crop_bottom = crop_bottom_override if type(crop_bottom_override) is int else crop_bottom
        crop_top = crop_top_override if type(crop_top_override) is int else crop_top
        
        crop_left = max(crop_left, 0)
        crop_right = max(crop_right, 0)
        crop_bottom = max(crop_bottom, 0)
        crop_top = max(crop_top, 0)
        
        if crop_left == 0 and crop_right == 0 and crop_bottom == 0 and crop_top == 0:
            return (images,)
        
        (count, original_y, original_x, colors) = images.size()
        cropped_images = images[
            :,
            crop_top:original_y-crop_bottom,
            crop_left:original_x-crop_right,
            :,
        ]#.clone()
        return (cropped_images,)

class CircularCrop:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "circular_crop"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "images": ("IMAGE",),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "radius": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "output_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            },
        }

    def circular_crop(self, images: torch.Tensor, center_x: float, center_y: float, radius: float, feather: int, output_size: int):
        batch_size, height, width, channels = images.shape
        
        # Create coordinate grids with higher precision
        y_coords = torch.linspace(0, 1, height, device=images.device)
        x_coords = torch.linspace(0, 1, width, device=images.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Adjust x coordinates to maintain circular shape
        x_grid_adjusted = (x_grid - center_x) * aspect_ratio + center_x
        
        # Calculate distance from center with higher precision
        dist = torch.sqrt((x_grid_adjusted - center_x)**2 + (y_grid - center_y)**2)
        
        # Create circular mask with higher precision
        mask = (dist <= radius).float()
        
        # Apply feathering if specified
        if feather > 0:
            feather_radius = feather / min(width, height)
            feather_mask = torch.clamp((radius - dist) / feather_radius, 0, 1)
            mask = mask * feather_mask
        
        # Expand mask for RGB channels
        mask_rgb = mask.unsqueeze(-1).unsqueeze(0).expand(batch_size, -1, -1, channels)
        
        # Apply mask to image
        result = images * mask_rgb
        
        # Find the exact boundaries of the circle
        y_indices = torch.where(mask > 0)[0]
        x_indices = torch.where(mask > 0)[1]
        
        if len(y_indices) > 0 and len(x_indices) > 0:
            min_y = y_indices.min().item()
            max_y = y_indices.max().item()
            min_x = x_indices.min().item()
            max_x = x_indices.max().item()
            
            # Calculate the size of the square crop
            crop_size = max(max_x - min_x + 1, max_y - min_y + 1)
            
            # Calculate the center of the crop
            center_y_crop = (min_y + max_y) // 2
            center_x_crop = (min_x + max_x) // 2
            
            # Calculate the square crop boundaries
            half_size = crop_size // 2
            min_y = max(0, center_y_crop - half_size)
            max_y = min(height - 1, center_y_crop + half_size)
            min_x = max(0, center_x_crop - half_size)
            max_x = min(width - 1, center_x_crop + half_size)
            
            # Ensure square crop
            crop_size = min(max_x - min_x + 1, max_y - min_y + 1)
            max_x = min_x + crop_size - 1
            max_y = min_y + crop_size - 1
            
            # Crop to square
            result = result[:, min_y:max_y+1, min_x:max_x+1, :]
            mask = mask[min_y:max_y+1, min_x:max_x+1]
        
        # Resize to square output size
        if output_size != width or output_size != height:
            # Resize to square
            result = torch.nn.functional.interpolate(
                result.permute(0, 3, 1, 2),  # Change to NCHW format
                size=(output_size, output_size),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # Change back to NHWC format
            
            # Resize mask
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            mask = torch.nn.functional.interpolate(
                mask,
                size=(output_size, output_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        
        # Create a transparent output with alpha channel (RGBA)
        if channels == 3:  # If RGB image
            # Create a new tensor with an additional alpha channel
            alpha_channel = mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)
            rgba = torch.cat([result, alpha_channel], dim=3)
            return (rgba,)
        else:
            # Just return the masked image if not RGB
            return (result,)

NODE_CLASS_MAPPINGS = {
    "ComfyUI_Image_Round__ImageRound": ImageRound,
    "ComfyUI_Image_Round__ImageRoundAdvanced": ImageRoundAdvanced,
    "ComfyUI_Image_Round__ImageCropAdvanced": ImageCropAdvanced,
    "ComfyUI_Image_Round__CircularCrop": CircularCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_Image_Round__ImageRound": "Round Image (Pad/Crop)",
    "ComfyUI_Image_Round__ImageRoundAdvanced": "Round Image (Pad/Crop) (Advanced)",
    "ComfyUI_Image_Round__ImageCropAdvanced": "Crop Image (Advanced)",
    "ComfyUI_Image_Round__CircularCrop": "Circular Crop",
}