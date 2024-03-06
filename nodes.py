class ImagePad:
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
    FUNCTION = "scale"
    OUTPUT_NODE = True
    CATEGORY = "image"

NODE_CLASS_MAPPINGS = {
    "ImagePad": ImagePad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePad": "Image Pad",
}