__version__ = "1.1"

from functools import total_ordering
from re import M
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar

class Sam3dBodyBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class Sam3dBody(desc.Node):
    category = "Mesh Generation"
    documentation = """This node computes a mesh from a monocular image using the Sam 3D Body deep model."""
    
    gpu = desc.Level.INTENSIVE

    size = avpar.DynamicViewsSize("input")
    parallelization = Sam3dBodyBlockSize()

    inputs = [
        desc.File(
            name="input",
            label="Input SfmData",
            description="Filepath of sfmData (.sfm or .abc) containing the filepaths of images to be processed.",
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
        # desc.ChoiceParam(
        #     name="device",
        #     label="Device",
        #     description="Model execution device",
        #     values=["cpu", "cuda"],
        #     value="cuda",
        #     exclusive=True,
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
        desc.File(
            name="overlay",
            label="Overlays",
            description="Reconstructed 3d mesh render.",
            semantic="image",
            value="{nodeCacheFolder}/<FILESTEM>_overlay.png",
            group="",
        ),
        desc.File(
            name="bbox",
            label="Bounding Boxes",
            description="Bounding boxes used by sam3d.",
            semantic="image",
            value="{nodeCacheFolder}/<FILESTEM>_bbox.png",
            group="",
        ),
    ]

    def preprocess(self, node):
        self.image_paths = get_image_paths_list(node.input.value)
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {node.input.value}')

    def processChunk(self, chunk):
        from sam3dBodyInference.utils import setup_sam_3d_body, process_image_with_mask, save_mesh_results

        import torch
        from img_proc import image
        import os
        from contextlib import nullcontext
        import numpy as np
        from pathlib import Path
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.input.value:
                chunk.logger.warning('No input sfmData given.')

            chunk_image_paths = self.image_paths[chunk.range.start:chunk.range.end]

            device = "cuda"
            if not torch.cuda.is_available():
                chunk.logger.error('CUDA is not available. Aborting...')

            # Initialize models
            chunk.logger.info("Loading SAM-3D-Body model...")
            checkpoint_path = os.getenv("SAM_3D_BODY_MODELS_PATH") + "model.ckpt"
            mhr_path = os.getenv("SAM_3D_BODY_MODELS_PATH") + "mhr_model.pt"
            # Set up SAM 3D Body estimator
            estimator = setup_sam_3d_body(checkpoint_path=checkpoint_path, mhr_path=mhr_path, detector_name=None, segmentor_path=None, device=device)

            # computation
            chunk.logger.info(f'Starting computation on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...')

            for idx, iFile in enumerate(chunk_image_paths):

                maskDirPath = Path(chunk.node.maskFolder.value)
                image_stem = Path(iFile).stem
                mask_file_name = str(image_stem) + "." + chunk.node.maskExtension.value
                iMask = os.path.join(maskDirPath, mask_file_name)

                img, h_ori, w_ori, PAR, orientation = image.loadImage(str(iFile), True)
                if img.shape[2]==4:
                    img_uint8 = (255.0 * img[:,:,:3]).astype(np.uint8)
                    img_mask = img[:,:,3] > 0
                else:
                    img_uint8 = (255.0 * img).astype(np.uint8)
                    mask, h_ori_mask, w_ori_mask, PAR_mask, orientation_mask = image.loadImage(str(iMask), True)
                    img_mask = mask > 0
                    if img_mask.shape[2] == 3:
                        img_mask = img_mask[..., -1]

                output = process_image_with_mask(estimator, img_uint8, img_mask)

                chunk.logger.info(f"Number of people detected: {len(output)}")
                # if len(output) != 0:
                #     chunk.logger.info(f"Output keys for first person: {list(output[0].keys())}")

                outputDirPath = Path(chunk.node.output.value)

                save_mesh_results(img_uint8[...,::-1], output, estimator.faces, outputDirPath, str(image_stem))

            chunk.logger.info('Sam3dBody end')
        finally:
            chunk.logManager.end()


def get_image_paths_list(input_path):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pathlib import Path

    if Path(input_path).suffix.lower() not in [".sfm", ".abc"] or not Path(input_path).exists():
        raise ValueError(f"Input path '{input_path}' is not a valid sfmData file path.")

    image_paths = []
    dataAV = sfmData.SfMData()
    if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
        views = dataAV.getViews()
        for id, v in views.items():
            image_paths.append(Path(v.getImage().getImagePath()))
    image_paths.sort()

    return image_paths
