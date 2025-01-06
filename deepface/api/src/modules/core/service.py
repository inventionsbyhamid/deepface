# built-in dependencies
import traceback
from typing import Optional, Union

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=broad-except


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


def analyze(
    img_path: Union[str, np.ndarray],
    actions: list,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = demographies
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400

def compute_embedding(
    img_path: Union[str, np.ndarray],
    model_name: str = "Facenet512",
    detector_backend: str = "fastmtcnn",
    enforce_detection: bool = False,
    align: bool = False,  # Align can be False since detector_backend is fastmtcnn
    anti_spoofing: bool = False,
):
    """
    Compute embeddings for all detected faces in an image.

    Args:
        img_path (str or np.ndarray): Path to the image or image array.
        model_name (str): The face recognition model to use.
        detector_backend (str): The backend to use for face detection.
        enforce_detection (bool): Whether to enforce face detection.
        align (bool): Whether to align the face.
        anti_spoofing (bool): Whether to perform anti-spoofing.

    Returns:
        dict: A dictionary containing embeddings and facial areas or error information.
    """
    try:
        result = []
        img_objs = DeepFace.extract_faces(
            img_path=img_path,
            enforce_detection=enforce_detection,
            detector_backend=detector_backend
        )

        if len(img_objs) == 0:
            return {"error": "No face detected"}, 400

        for img_obj in img_objs:
            try:
                facial_area = img_obj.get("facial_area")
                if facial_area is None:
                    logger.error("Facial area not found in the detected face object.")
                    continue  # Skip to the next face if facial_area is missing

                logger.debug(f"Face found in: {facial_area}")

                img_content = img_obj.get("face")
                if img_content is None:
                    logger.error("Face content is missing.")
                    continue  # Skip to the next face if face content is missing

                # Generate embedding for the face
                embedding_objs = DeepFace.represent(
                    img_path=img_content,
                    model_name=model_name,
                    enforce_detection=enforce_detection,
                    detector_backend="skip"  # Skip detection as faces are already extracted
                )

                if not embedding_objs:
                    logger.error("Failed to generate embedding for the face.")
                    continue  # Skip if embedding generation failed

                img_representation = embedding_objs[0].get("embedding")
                if img_representation is None:
                    logger.error("Embedding not found in the representation object.")
                    continue  # Skip if embedding is missing

                result.append({
                    "embedding": img_representation,
                    "facial_area": facial_area
                })

            except Exception as face_error:
                logger.error(f"Error processing a face: {face_error}")
                # Continue processing other faces instead of exiting

        if not result:
            return {"error": "No embeddings were generated."}, 400

        return {"results": result}, 200

    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while computing embeddings: {str(err)} - {tb_str}"}, 400
