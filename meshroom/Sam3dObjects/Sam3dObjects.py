__version__ = "1.1"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar

class Sam3dObjectsBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute("blockSize").value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute("blockSize").value)))
            return node.attribute("blockSize").value, size, nbBlocks
        else:
            return size, size, 1


class Sam3dObjects(desc.Node):
    """
    This node computes a mesh from a monocular image using the Sam 3D Objects deep model.
    """
    category = "Mesh Generation"
    
    gpu = desc.Level.EXTREME

    size = avpar.DynamicViewsSize("input")
    parallelization = Sam3dObjectsBlockSize()

    inputs = [
        desc.File(
            name="input",
            label="Input SfmData",
            description="Filepath of SfMData (.sfm or .abc) containing the filepaths of images to be processed.",
            value="",
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
            name="output",
            label="Output Folder",
            description="Output folder containing the computed meshes.",
            value="{nodeCacheFolder}",
        ),
    ]

    def get_image_paths(self, node):
        self.image_paths = get_image_paths_list(node.input.value)
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No image files found in {node.input.value}.")

    def processChunk(self, chunk):
        from sam3dObjectsInference import inference

        import torch
        from img_proc import image
        import os
        import numpy as np
        from pathlib import Path

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.input.value:
                chunk.logger.warning("No input SfMData given.")

            self.get_image_paths(chunk.node)
            chunk_image_paths = self.image_paths[chunk.range.start:chunk.range.end]

            device = "cuda"
            if not torch.cuda.is_available():
                chunk.logger.error("CUDA is not available. Aborting...")

            # Initialize models
            chunk.logger.info("Loading SAM-3D-Objects model...")
            config_path = os.getenv("SAM_3D_OBJECTS_MODELS_PATH") + "/pipeline.yaml"
            inference = inference.Inference(config_path, compile=False)

            # computation
            chunk.logger.info(f"Starting computation on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...")

            for _, iFile in enumerate(chunk_image_paths):
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
                    chunk.logger.warning(f"No object segmented in {Path(iFile).name}. Skipping 3D object reconstruction...")
                    continue

                output = inference(img_uint8, img_mask, seed=42)

                outputDirPath = Path(chunk.node.output.value)

                mesh_file_name = "mesh_" + str(image_stem) + ".obj"
                out_mesh_path = os.path.join(outputDirPath, mesh_file_name)
                output["glb"].export(out_mesh_path, include_normals=True)

            chunk.logger.info("Sam3dObjects end")
        finally:
            chunk.logManager.end()


def get_image_paths_list(input_path):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pathlib import Path

    if Path(input_path).suffix.lower() not in [".sfm", ".abc"] or not Path(input_path).exists():
        raise ValueError(f"Input path '{input_path}' is not a valid SfMData file path.")

    image_paths = []
    dataAV = sfmData.SfMData()
    if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
        views = dataAV.getViews()
        for id, v in views.items():
            image_paths.append(Path(v.getImage().getImagePath()))
    image_paths.sort()

    return image_paths
