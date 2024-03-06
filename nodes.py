class ImageRoundPad:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s):
        return True

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "pad_rounded"
    OUTPUT_NODE = True
    CATEGORY = "image"

NODE_CLASS_MAPPINGS = {
    "ImageRoundPad": ImageRoundPad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRoundPad": "Image Pad (Rounded)",
}