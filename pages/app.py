from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from pages.functions import create_drive_service, upload_file_to_drive

ASSETS_DIR = Path("assets")
DETECTION_MODEL_PATH = ASSETS_DIR / "retina_detector.pt"
CLASSIFICATION_MODEL_PATH = ASSETS_DIR / "retina_classifier.onnx"
DEFAULT_UPLOAD_FOLDER_ID = "1ZJl54Kl-kpfPY6WbbSW29YTmDxkejN1F"
CLASSIFICATION_LABELS = {
    "HM": "Hemorrhage",
    "dr_1": "DR (Stage 1)",
    "dr_2": "DR (Stage 2)",
    "dr_3": "DR (Stage 3)",
    "dr_4": "DR (Stage 4)",
    "hard_exudates": "Hard Exudates",
    "cotton_wool_spot": "Soft Exudates",
    "venous_looping": "Venous Looping",
    "viterous_HM": "Vitreous Hemorrhage",
    "PreRetinal_HM": "Preretinal Hemorrhage",
    "VB": "Venous Beading",
    "IRMA": "Intraretinal Microvascular Abnormality",
    "FPD": "Fibrous-Proliferation (Disk)",
    "FPE": "Fibrous-Proliferation (Elsewhere)",
    "NVD": "Neovascularization (Disk)",
    "NVE": "Neovascularization (Elsewhere)",
    "FTMH": "Full Thickness Macular Hole",
    "ME": "Macular Edema",
    "ERM": "Epiretinal Membrane",
    "laser": "Laser",
    "microaneurysms": "Microaneurysms",
    "no_dr": "No DR",
    "no_disease": "No Disease",
}


class ONNXModel:
    def __init__(self, model_path: str | Path):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def predict(self, image: np.ndarray):
        return self.session.run(self.output_names, {self.input_name: image})


@st.cache_resource(show_spinner=False)
def load_models():
    det_model = YOLO(str(DETECTION_MODEL_PATH))
    cls_model = ONNXModel(CLASSIFICATION_MODEL_PATH)
    return det_model, cls_model


@st.cache_resource(show_spinner=False)
def get_drive_service():
    return create_drive_service()


def get_image(file_path: str | Path):
    image = Image.open(file_path).convert("RGB").resize((500, 500))
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def get_prediction_tags(all_pred):
    prediction_tags = {}
    dr_prediction = all_pred[0][0]

    def score(value):
        return float(value)

    if dr_prediction[0] >= 0.25:
        prediction_tags["microaneurysms"] = score(dr_prediction[0])
    if dr_prediction[1] >= 0.25:
        prediction_tags["hard_exudates"] = score(dr_prediction[1])
    if dr_prediction[2] >= 0.25:
        prediction_tags["cotton_wool_spot"] = score(dr_prediction[2])
    if dr_prediction[3] >= 0.25:
        prediction_tags["DME"] = score(dr_prediction[3])
    if dr_prediction[4] >= 0.25:
        prediction_tags["venous_looping"] = score(dr_prediction[4])
    if dr_prediction[5] >= 0.25:
        prediction_tags["FPD"] = score(dr_prediction[5])
    if dr_prediction[6] >= 0.25:
        prediction_tags["FPE"] = score(dr_prediction[6])
    if dr_prediction[7] >= 0.25:
        prediction_tags["NVD"] = score(dr_prediction[7])
    if dr_prediction[8] >= 0.25:
        prediction_tags["NVE"] = score(dr_prediction[8])
    if dr_prediction[9] >= 0.25:
        prediction_tags["viterous_HM"] = score(dr_prediction[9])
    if dr_prediction[10] >= 0.25:
        prediction_tags["PreRetinal_HM"] = score(dr_prediction[10])
    if dr_prediction[11] >= 0.25:
        prediction_tags["laser"] = score(dr_prediction[11])
    if all_pred[1][0][0] >= 0.25:
        prediction_tags["HM"] = score(all_pred[1][0][0])
    if all_pred[1][0][1] >= 0.25:
        prediction_tags["VB"] = score(all_pred[1][0][1])
    if all_pred[1][0][2] >= 0.25:
        prediction_tags["IRMA"] = score(all_pred[1][0][2])
    if all_pred[3][0][0] >= 0.25:
        prediction_tags["no_dr"] = score(all_pred[3][0][0])

    if all_pred[2][0][0] >= 0.5:
        prediction_tags["ERM"] = score(all_pred[2][0][0])
    if all_pred[2][0][1] >= 0.5:
        prediction_tags["Myopic degeneration"] = score(all_pred[2][0][1])
    if all_pred[2][0][2] >= 0.5:
        prediction_tags["Drusen"] = score(all_pred[2][0][2])
    if all_pred[2][0][3] >= 0.5:
        prediction_tags["Hypertensive Retinopathy"] = score(all_pred[2][0][3])
    if all_pred[2][0][4] >= 0.5:
        prediction_tags["Peripapilary Atrophy"] = score(all_pred[2][0][4])
    if all_pred[2][0][5] >= 0.5:
        prediction_tags["Tortuous vessels"] = score(all_pred[2][0][5])
    if all_pred[2][0][6] >= 0.5:
        prediction_tags["Macular Dystrophy"] = score(all_pred[2][0][6])
    if all_pred[2][0][7] >= 0.5:
        prediction_tags["Chorioretinitis"] = score(all_pred[2][0][7])
    if all_pred[2][0][8] >= 0.5:
        prediction_tags["Papilledema"] = score(all_pred[2][0][8])
    if all_pred[2][0][9] >= 0.5:
        prediction_tags["Nevus"] = score(all_pred[2][0][9])
    if all_pred[2][0][10] >= 0.5:
        prediction_tags["FTMH"] = score(all_pred[2][0][10])
    if all_pred[2][0][11] >= 0.5:
        prediction_tags["CRVO"] = score(all_pred[2][0][11])
    if all_pred[2][0][12] >= 0.5:
        prediction_tags["BRVO"] = score(all_pred[2][0][12])

    if len(all_pred) > 4 and all_pred[4][0][0] >= 0.5:
        prediction_tags["no_disease"] = score(all_pred[4][0][0])
    if all_pred[3][0][1] >= 0.25:
        prediction_tags["dr_1"] = score(all_pred[3][0][1])
    if all_pred[3][0][2] >= 0.25:
        prediction_tags["dr_2"] = score(all_pred[3][0][2])
    if all_pred[3][0][3] >= 0.25:
        prediction_tags["dr_3"] = score(all_pred[3][0][3])
    if all_pred[3][0][4] >= 0.25:
        prediction_tags["dr_4"] = score(all_pred[3][0][4])

    extra_labels = ["Myelination", "Glaucoma", "Cataract", "ME", "ARMD"]
    for offset, label in enumerate(extra_labels, start=13):
        if len(all_pred[2][0]) > offset and all_pred[2][0][offset] >= 0.5:
            prediction_tags[label] = score(all_pred[2][0][offset])

    return prediction_tags


def get_dr_level(pred_dict):
    all_disease = [
        "microaneurysms",
        "hard_exudates",
        "cotton_wool_spot",
        "DME",
        "venous_looping",
        "FPD",
        "FPE",
        "NVD",
        "NVE",
        "viterous_HM",
        "PreRetinal_HM",
        "laser",
        "HM",
        "VB",
        "IRMA",
        "ERM",
        "Myopic degeneration",
        "Drusen",
        "Hypertensive Retinopathy",
        "Peripapilary Atrophy",
        "Tortuous vessels",
        "Macular Dystrophy",
        "Chorioretinitis",
        "Papilledema",
        "Nevus",
        "FTMH",
        "CRVO",
        "BRVO",
        "Myelination",
        "Glaucoma",
        "Cataract",
        "ME",
        "ARMD",
        "dr_1",
        "dr_2",
        "dr_3",
        "dr_4",
    ]
    all_keys = [
        "microaneurysms",
        "hard_exudates",
        "cotton_wool_spot",
        "DME",
        "venous_looping",
        "FPD",
        "FPE",
        "NVD",
        "NVE",
        "viterous_HM",
        "PreRetinal_HM",
        "laser",
        "HM",
        "VB",
        "IRMA",
        "no_dr",
        "no_disease",
        "dr_1",
        "dr_2",
        "dr_3",
        "dr_4",
    ]
    dr_4_keys = ["laser", "FPD", "FPE", "NVD", "NVE", "viterous_HM", "PreRetinal_HM"]
    dr_3_keys = ["VB", "IRMA"]
    dr_2_keys = ["hard_exudates", "cotton_wool_spot", "venous_looping"]

    normalized = pred_dict.copy()
    for key in set(all_keys + all_disease):
        normalized.setdefault(key, -1)

    hm_conf = normalized["HM"]
    dr_2_keys_maxconf = normalized[max(dr_2_keys, key=lambda item: normalized[item])]
    dr_3_keys_maxconf = normalized[max(dr_3_keys, key=lambda item: normalized[item])]
    dr_4_keys_maxconf = normalized[max(dr_4_keys, key=lambda item: normalized[item])]
    disease_maxconf = normalized[max(all_disease, key=lambda item: normalized[item])]

    if normalized["no_disease"] >= 0.85:
        disease_pred = "no_disease"
        grading_pred = "dr_0"
        mtm_vtdr = "non-VTDR"
    else:
        if normalized["dr_4"] > 0.45 or dr_4_keys_maxconf >= 0.45 or normalized["laser"] > 0.3:
            grading_pred = "dr_4"
            mtm_vtdr = "vtdr"
            disease_pred = "disease"
        elif normalized["dr_3"] > 0.45 or dr_3_keys_maxconf >= 0.35 or hm_conf >= 0.75:
            grading_pred = "dr_3"
            mtm_vtdr = "vtdr"
            disease_pred = "disease"
        elif normalized["dr_2"] > 0.45 or dr_2_keys_maxconf >= 0.45 or hm_conf > 0.4:
            grading_pred = "dr_2"
            mtm_vtdr = "non-VTDR"
            disease_pred = "disease"
        elif (
            normalized["dr_1"] > normalized["no_dr"]
            and normalized["dr_1"] > normalized["no_disease"]
        ) or (normalized["dr_1"] > 0.45 or normalized["microaneurysms"] >= 0.4):
            grading_pred = "dr_1"
            mtm_vtdr = "non-VTDR"
            disease_pred = "disease"
        elif normalized["no_dr"] > 0.5:
            grading_pred = "dr_0"
            mtm_vtdr = "non-VTDR"
            disease_pred = "disease" if disease_maxconf >= 0.5 else "no_disease"
        elif normalized["no_disease"] >= 0.5:
            disease_pred = "review" if disease_maxconf >= 0.5 else "no_disease"
            grading_pred = "dr_0"
            mtm_vtdr = "non-VTDR"
        else:
            disease_pred = "disease"
            grading_pred = "review"
            mtm_vtdr = "review"

        if min(normalized["DME"], normalized["hard_exudates"]) >= 0.25 and (
            normalized["DME"] + normalized["hard_exudates"]
        ) >= 0.8:
            mtm_vtdr = "vtdr"

    return mtm_vtdr, grading_pred, disease_pred


def cls_predict(img_path, cls_model, return_dr_level=True):
    prediction_tags = cls_model.predict(get_image(img_path))
    final_pred_tags = get_prediction_tags(prediction_tags)
    mtm_vtdr, grading_pred, disease_pred = "", "", ""
    if return_dr_level:
        mtm_vtdr, grading_pred, disease_pred = get_dr_level(final_pred_tags.copy())
    return final_pred_tags, mtm_vtdr, grading_pred, disease_pred


def transform_classification_result(cls_result_dict):
    return {
        "diseases": cls_result_dict[0],
        "vision_threatening": cls_result_dict[1],
        "dr_grade": cls_result_dict[2],
        "disease_present": cls_result_dict[3],
    }


def generate_colors(num_colors):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=int)
    return {index: tuple(color) for index, color in enumerate(colors)}


def predict(image_path, det_model, cls_model, save=True, conf_threshold=0.25):
    cls_result_dict = transform_classification_result(cls_predict(image_path, cls_model))
    results = det_model.predict(
        str(image_path),
        imgsz=800,
        show_labels=False,
        show_conf=False,
        save=False,
        conf=conf_threshold,
        verbose=False,
    )

    image = cv2.imread(str(image_path))
    class_names = det_model.names
    class_colors = generate_colors(len(class_names))
    detected_classes = {}
    class_serial_mapping = {}
    serial_number = 1
    result_dict = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls)

            if class_id not in class_serial_mapping:
                class_serial_mapping[class_id] = serial_number
                detected_classes[serial_number] = class_names[class_id]
                serial_number += 1

            class_number = class_serial_mapping[class_id]
            result_dict[class_names[class_id]] = {
                "bounding_box": [x1, y1, x2, y2],
                "confidence": conf,
            }

            if save:
                color = tuple(map(int, class_colors[class_id]))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                text = str(class_number)
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                )
                cv2.rectangle(
                    image,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width + 10, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    image,
                    text,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

    output_image_path = None
    if save and image is not None and detected_classes:
        legend_height = max(60, 40 * len(detected_classes) + 20)
        legend_width = image.shape[1]
        legend_background = np.full(
            (legend_height, legend_width, 3),
            (255, 255, 255),
            dtype=np.uint8,
        )
        cv2.rectangle(legend_background, (0, 0), (legend_width, legend_height), (0, 0, 0), 2)

        for index, (serial, class_name) in enumerate(detected_classes.items()):
            text = f"{serial}. {class_name}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )
            legend_x, legend_y = 10, 40 * (index + 1)
            class_id = next(key for key, value in class_serial_mapping.items() if value == serial)
            color = tuple(map(int, class_colors[class_id]))
            cv2.rectangle(
                legend_background,
                (legend_x - 5, legend_y - text_height - 5),
                (legend_x + text_width + 5, legend_y + baseline),
                color,
                -1,
            )
            cv2.putText(
                legend_background,
                text,
                (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        output_image = np.concatenate((image, legend_background), axis=0)
        output_image_path = Path(tempfile.gettempdir()) / f"fundus_result_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(output_image_path), output_image)
    elif save and image is not None:
        output_image_path = Path(tempfile.gettempdir()) / f"fundus_result_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(output_image_path), image)

    return output_image_path, result_dict, cls_result_dict


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    temp_path = Path(tempfile.gettempdir()) / f"fundus_upload_{uuid.uuid4().hex}{suffix}"
    temp_path.write_bytes(uploaded_file.getbuffer())
    return temp_path


def render_result_details(classification_details, detection_details):
    diseases = classification_details.get("diseases", {})
    vision_threatening = classification_details.get("vision_threatening", "Absent")
    disease_present = classification_details.get("disease_present", "Absent")
    transformed_diseases = [CLASSIFICATION_LABELS.get(name, name) for name in diseases.keys()]

    st.markdown("**Eye Status**")
    st.write(disease_present.capitalize())

    st.markdown("**Vision Threatening**")
    vision_status = "Yes" if vision_threatening == "vtdr" else "No"
    st.write(vision_status)

    st.markdown(
        """
        <style>
        .custom-container {
            background-color: #f9f9f9;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            max-height: 150px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Diseases Present**")
    diseases_text = "<br>".join(transformed_diseases) if transformed_diseases else "None"
    st.markdown(f'<div class="custom-container">{diseases_text}</div>', unsafe_allow_html=True)

    with st.expander("Classification JSON"):
        st.json(classification_details)
    with st.expander("Detection JSON"):
        st.json(detection_details)


def run_prediction_flow(uploaded_file, conf_threshold):
    det_model, cls_model = load_models()
    image_path = save_uploaded_file(uploaded_file)
    service = get_drive_service()
    upload_status = "disabled"

    if service is not None:
        uploaded_file_id = upload_file_to_drive(
            service=service,
            file_path=image_path,
            file_name=uploaded_file.name,
            mime_type=uploaded_file.type or "image/jpeg",
            folder_id=DEFAULT_UPLOAD_FOLDER_ID,
        )
        upload_status = "success" if uploaded_file_id else "failed"

    saved_path, detection_details, classification_details = predict(
        image_path=image_path,
        det_model=det_model,
        cls_model=cls_model,
        save=True,
        conf_threshold=conf_threshold,
    )
    return saved_path, detection_details, classification_details, upload_status


st.markdown(
    """
    <style>
    .main-content {
        padding-top: 50px;
    }
    .stSlider {
        margin-bottom: 25px;
    }
    .stSlider > div {
        height: 60px;
        padding: 10px;
        background-color: #f4f4f4;
        border: 1px solid #ccc;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    _, upload_col, _ = st.columns([1, 3, 1])
    with upload_col:
        st.write("Upload a Fundus image to get diagnosis")
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "tiff"],
            help="Upload your retinal image here.",
        )
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            help="Adjust the threshold for object detection confidence.",
        )

    st.session_state.setdefault("detection_done", False)
    st.session_state.setdefault("feedback_given", False)
    st.session_state.setdefault("prediction_result", None)
    st.session_state.setdefault("prediction_input_name", None)

    if uploaded_file is None:
        st.session_state.prediction_result = None
        st.session_state.prediction_input_name = None
        return

    if st.session_state.prediction_input_name != uploaded_file.name:
        st.session_state.prediction_result = None
        st.session_state.prediction_input_name = uploaded_file.name

    _, button_col, _ = st.columns([1, 3, 1])
    with button_col:
        if st.button("Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Running prediction..."):
                saved_path, detection_details, classification_details, upload_status = run_prediction_flow(
                    uploaded_file,
                    conf_threshold,
                )

            st.session_state.prediction_result = {
                "saved_path": str(saved_path) if saved_path else None,
                "detection_details": detection_details,
                "classification_details": classification_details,
                "upload_status": upload_status,
                "confidence_threshold": conf_threshold,
            }

            if upload_status == "failed":
                st.toast(
                    "Prediction completed, but Google Drive upload failed. The app continued locally.",
                    icon="⚠️",
                )

            st.session_state.detection_done = True
            st.session_state.feedback_given = False

    prediction_result = st.session_state.prediction_result
    if prediction_result:
        _, result_col, _ = st.columns([1, 3, 1])
        with result_col:
            image_col, detail_col = st.columns([2, 1])
            with image_col:
                if prediction_result["saved_path"]:
                    st.image(
                        prediction_result["saved_path"],
                        caption="Detected Objects with Bounding Boxes",
                        use_container_width=True,
                    )

            with detail_col:
                render_result_details(
                    prediction_result["classification_details"],
                    prediction_result["detection_details"],
                )


main()
