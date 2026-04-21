from fastapi import HTTPException
from ultralytics import YOLO
from app.utils.config import MODEL_URL

class Inference:
    def __init__(self):
        pass

    _model : YOLO | None = None

    def _get_model(self) -> YOLO:
        if Inference._model is None:
            if not MODEL_URL.is_file():
                raise HTTPException(
                    status_code=503,
                    detail=f"Model weights not found at {MODEL_URL}. Place best.pt in the project root or set YOLO_MODEL_PATH."
                )
            Inference._model = YOLO(str(MODEL_URL))
        return Inference._model


    def model_prediction(
        self, 
        image_path: str
    ):
        try:
            model = self._get_model()
            results = model.predict(
                source=image_path,
                conf=0.4,
                iou=0.45,
                task='detect'
            )
        except Exception as e:
            import traceback
            traceback.print_exc() # This prints the REAL error to your console
            raise HTTPException(status_code=500, detail=str(e))
        
        return results[0]