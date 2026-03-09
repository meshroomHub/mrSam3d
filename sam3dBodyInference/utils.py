"""
Utility functions for SAM 3D Body demo notebook
"""

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body, load_sam_3d_body_hf, SAM3DBodyEstimator
from sam_3d_body.visualization.renderer import Renderer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def setup_sam_3d_body(
    hf_repo_id: str = "facebook/sam-3d-body-vith",
    checkpoint_path: str = "",
    mhr_path: str = "",
    detector_name: str = "vitdet",
    segmentor_name: str = "sam2",
    fov_name: str = "moge2",
    detector_path: str = "",
    segmentor_path: str = "",
    fov_path: str = "",
    device: str = "cuda",
):
    """
    Set up SAM 3D Body estimator with optional components.

    Args:
        hf_repo_id: HuggingFace repository ID for the model
        detector_name: Name of detector to use (default: "vitdet")
        segmentor_name: Name of segmentor to use (default: "sam2")
        fov_name: Name of FOV estimator to use (default: "moge2")
        detector_path: URL or path for human detector model
        segmentor_path: Path to human segmentor model (optional)
        fov_path: path for FOV estimator
        device: Device to use (default: auto-detect cuda/cpu)

    Returns:
        estimator: SAM3DBodyEstimator instance ready for inference
    """

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load core model from HuggingFace
    if checkpoint_path != "":
        model, model_cfg = load_sam_3d_body(checkpoint_path=checkpoint_path, mhr_path=mhr_path, device=device)
    else:
        print(f"Loading SAM 3D Body model from {hf_repo_id}...")
        model, model_cfg = load_sam_3d_body_hf(hf_repo_id, device=device)

    # Initialize optional components
    human_detector, human_segmentor, fov_estimator = None, None, None

    if detector_name:
        print(f"Loading human detector from {detector_name}...")
        from sam3dBodyInference.tools.build_detector import HumanDetector

        human_detector = HumanDetector(name=detector_name, device=device)

    if segmentor_path:
        print(f"Loading human segmentor from {segmentor_path}...")
        from sam3dBodyInference.tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=segmentor_name, device=device, path=segmentor_path
        )

    if fov_name:
        print(f"Loading FOV estimator from {fov_name}...")
        from sam3dBodyInference.tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=fov_name, device=device)

    # Create estimator wrapper
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    print(f"Setup complete!")
    # print(
    #     f"  Human detector: {'✓' if human_detector else '✗ (will use full image or manual bbox)'}"
    # )
    # print(
    #     f"  Human segmentor: {'✓' if human_segmentor else '✗ (mask inference disabled)'}"
    # )
    print(f"  FOV estimator: {'✓' if fov_estimator else '✗ (will use default FOV)'}")

    return estimator


def save_mesh_results(
    img: np.ndarray,
    outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    save_dir: str,
    image_name: str,
) -> List[str]:
    """Save 3D mesh results to files"""
    import json

    os.makedirs(save_dir, exist_ok=True)

    # Save focal length
    if outputs:
        focal_length_data = {"focal_length": float(outputs[0]["focal_length"])}
        focal_length_path = os.path.join(save_dir, f"{image_name}_focal_length.json")
        with open(focal_length_path, "w") as f:
            json.dump(focal_length_data, f, indent=2)
        print(f"Saved focal length: {focal_length_path}")

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # Store individual mesh
        tmesh = renderer.vertices_to_trimesh(
            person_output["pred_vertices"], person_output["pred_cam_t"], LIGHT_BLUE
        )

        suffix = f"_{pid:03d}" if len(outputs) > 1 else ''

        mesh_filename = f"{image_name}_mesh{suffix}.obj"
        mesh_path = os.path.join(save_dir, mesh_filename)
        tmesh.export(mesh_path, include_normals=True)

        # Save individual overlay image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        overlay_filename = f"{image_name}_overlay{suffix}.png"
        cv2.imwrite(os.path.join(save_dir, overlay_filename), img_mesh_overlay)

        # Save bbox image
        img_bbox = img.copy()
        bbox = person_output["bbox"]
        img_bbox = cv2.rectangle(
            img_bbox,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            4,
        )
        bbox_filename = f"{image_name}_bbox{suffix}.png"
        cv2.imwrite(os.path.join(save_dir, bbox_filename), img_bbox)

        print(f"Saved mesh: {mesh_path}")
        print(f"Saved overlay: {os.path.join(save_dir, overlay_filename)}")
        print(f"Saved bbox: {os.path.join(save_dir, bbox_filename)}")


def process_image_with_mask(estimator, image: str, mask):
    """
    Process image with external mask input.

    Note: The refactored code requires bboxes to be provided along with masks.
    This function automatically computes bboxes from the mask.
    """

    # Ensure mask is binary (0 or 255)
    mask_binary = mask.astype(np.uint8) * 255

    print(f"Processing image with external mask")
    # print(f"Mask shape: {mask_binary.shape}, unique values: {np.unique(mask_binary)}")

    # Compute bounding box from mask (required by refactored code)
    # Find all non-zero pixels in the mask
    coords = cv2.findNonZero(mask_binary)
    if coords is None:
        print("Warning: Mask is empty, no objects detected")
        return []

    # Get bounding box from mask contours
    x, y, w, h = cv2.boundingRect(coords)
    bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)

    print(f"Computed bbox from mask: {bbox[0]}")

    # Process with external mask and computed bbox
    # Note: The mask needs to match the number of bboxes (1 bbox -> 1 mask)
    outputs = estimator.process_one_image(image, bboxes=bbox, masks=mask_binary)

    return outputs
