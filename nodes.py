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

class ImageRound:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "round_image"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
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

    @classmethod
    def VALIDATE_INPUTS(s):
        return True

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
        if nearest_x == 1 and nearest_y == 1:
            return (images,)
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
            return (images,)
        left, right = split_value(delta_x, split_x)
        bottom, top = split_value(delta_y, split_y)
        
        if round_up:
            padding = (0, 0, left, right, top, bottom)
            memory_format = None
            if images.is_contiguous():
                memory_format = torch.contiguous_format
            else:
                raise NotImplementedError("Unknown memory format!")
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
                return (new_images,)
        else:
            cropped_images = images[
                :,
                abs(bottom):original_y-abs(top),
                abs(left):original_x-abs(right),
                :,
            ]#.clone()
            return (cropped_images,)

NODE_CLASS_MAPPINGS = {
    "ImageRound": ImageRound,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRound": "Round Image (Pad/Crop)",
}