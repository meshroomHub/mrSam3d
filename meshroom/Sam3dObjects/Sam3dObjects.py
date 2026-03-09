__version__ = "1.0"

from functools import total_ordering
from re import M
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class Sam3dObjectsNodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        from pathlib import Path
        import itertools

        input_path_param = node.attribute(self._params[0])
        extension_param = node.attribute(self._params[1])

        input_path = input_path_param.value
        extension = extension_param.value
        include_suffixes = [extension.lower(), extension.upper()]

        size = 1
        if Path(input_path).is_dir():
            image_paths = list(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
            size = len(image_paths)
        elif node.attribute(self._params[0]).isLink:
            size = node.attribute(self._params[0]).inputLink.node.size
        
        return size


class Sam3dObjectsBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class Sam3dObjects(desc.Node):
    category = "Mesh Generation"
    documentation = """This node computes a mesh from a monocular image using the Sam 3D Objects deep model."""
    
    gpu = desc.Level.EXTREME

    size = Sam3dObjectsNodeSize(['inputImages', 'inputExtension'])
    parallelization = Sam3dObjectsBlockSize()

    inputs = [
        desc.File(
            name="inputImages",
            label="Input Images",
            description="Input images to estimate the depth from. Folder path or sfmData filepath",
            value="",
        ),
        desc.ChoiceParam(
            name="inputExtension",
            label="Input Extension",
            description="Extension of the input images. This will be used to determine which images are to be used if \n"
                        "a directory is provided as the input.",
            values=["jpg", "jpeg", "png", "exr"],
            value="png",
            exclusive=True,
        ),
        desc.File(
            name="maskFolder",
            label="Mask Folder",
            description="Folder containing input masks named like images. Optional if images have an alpha channel.",
            value="",
        ),
        desc.ChoiceParam(
            name="maskExtension",
            label="Mask Extension",
            description="Extension of the input masks.",
            values=["jpg", "jpeg", "png", "exr"],
            value="png",
            exclusive=True,
        ),
        # desc.IntParam(
        #     name="textureResolution",
        #     label="Texture Resolution",
        #     value=1024,
        #     description="Texture atlas resolution. Default: 1024",
        #     range=(256, 4096, 64),
        # ),
        # desc.FloatParam(
        #     name="foregroundRatio",
        #     label="Foreground Ratio",
        #     value=1.3,
        #     description="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 1.3",
        #     range=(0.5, 2.0, 0.01),
        # ),
        # desc.ChoiceParam(
        #     name="remeshOption",
        #     label="Remesh Option",
        #     description="Remeshing option",
        #     values=["none", "triangle", "quad"],
        #     value="none",
        #     exclusive=True,
        # ),
        # desc.ChoiceParam(
        #     name="reductionCountType",
        #     label="Reduction Count Type",
        #     description="Vertex count type",
        #     values=["keep", "vertex", "faces"],
        #     value="keep",
        #     exclusive=True,
        # ),
        # desc.IntParam(
        #     name="targetCount",
        #     label="Target Count",
        #     value=2000,
        #     description="Selected target count.",
        #     range=(100, 10000, 100),
        # ),
        # desc.ChoiceParam(
        #     name="device",
        #     label="Device",
        #     description="Model execution device",
        #     values=["cpu", "cuda"],
        #     value="cuda",
        #     exclusive=True,
        # ),
        # desc.BoolParam(
        #     name="lowVramMode",
        #     label="Low Vram Mode",
        #     value=False,
        #     description="Use low VRAM mode. Sam3dObjects consumes 10.5GB of VRAM by default. "
        #                 "This mode will reduce the VRAM consumption to roughly 7GB but in exchange "
        #                 "the model will be slower. Default: False",
        #     enabled=lambda node: node.device.value == "cuda",
        # ),
        desc.IntParam(
            name="blockSize",
            label="Block Size",
            value=50,
            description="Sets the number of images to process in one chunk. If set to 0, all images are processed at once.",
            range=(0, 1000, 1),
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name='output',
            label='Output Folder',
            description="Output folder containing the computed meshes.",
            value="{nodeCacheFolder}",
        ),
    ]

    def preprocess(self, node):
        extension = node.inputExtension.value
        input_path = node.inputImages.value

        image_paths = get_image_paths_list(input_path, extension)

        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')

        self.image_paths = image_paths

    def processChunk(self, chunk):
        from sam3dObjectsInference import inference

        import torch
        from img_proc import image
        import os
        from contextlib import nullcontext
        import numpy as np
        from pathlib import Path
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            if not chunk.node.inputImages.value:
                chunk.logger.warning('No input folder given.')

            chunk_image_paths = self.image_paths[chunk.range.start:chunk.range.end]

            device = "cuda"
            if not torch.cuda.is_available():
                chunk.logger.error('CUDA is not available. Aborting...')

            # Initialize models
            chunk.logger.info("Loading SAM-3D-Objects model...")
            config_path = os.getenv("SAM_3D_OBJECTS_MODELS_PATH") + "/pipeline.yaml"
            inference = inference.Inference(config_path, compile=False)

            # computation
            chunk.logger.info(f'Starting computation on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...')

            for idx, iFile in enumerate(chunk_image_paths):

                maskDirPath = Path(chunk.node.maskFolder.value)
                image_stem = Path(iFile).stem
                mask_file_name = str(image_stem) + "." + chunk.node.maskExtension.value
                iMask = os.path.join(maskDirPath, mask_file_name)

                img, h_ori, w_ori, PAR, orientation = image.loadImage(str(iFile), True)
                if img.shape[2] == 4:
                    img_uint8 = (255.0 * img[:,:,:3]).astype(np.uint8)
                    img_mask = img[:,:,3] > 0
                else:
                    img_uint8 = (255.0 * img).astype(np.uint8)
                    mask, h_ori_mask, w_ori_mask, PAR_mask, orientation_mask = image.loadImage(str(iMask), True)
                    img_mask = mask > 0
                    if img_mask.shape[2] == 3:
                        img_mask = img_mask[..., -1]

                if img_mask.max() != True:
                    chunk.logger.warning(f"No object segmented in {Path(iFile).name}. Skipping 3d object reconstruction...")
                    continue

                output = inference(img_uint8, img_mask, seed=42)

                outputDirPath = Path(chunk.node.output.value)

                mesh_file_name = "mesh_" + str(image_stem) + ".obj"
                out_mesh_path = os.path.join(outputDirPath, mesh_file_name)
                output["glb"].export(out_mesh_path, include_normals=True)

            chunk.logger.info('Sam3dObjects end')
        finally:
            chunk.logManager.end()


def get_image_paths_list(input_path, extension):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pathlib import Path
    import itertools

    include_suffixes = [extension.lower(), extension.upper()]
    image_paths = []

    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
    elif Path(input_path).suffix.lower() in [".sfm", ".abc"]:
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                    image_paths.append(Path(v.getImage().getImagePath()))
            image_paths.sort()
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths
