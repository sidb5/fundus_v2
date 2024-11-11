import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import onnxruntime as ort
from pages.functions import create_drive_service, upload_file_to_drive
import time
import uuid
from pages.config_loader import load_config

class ONNXModel:
    def __init__(self, model_path):
        # Load the ONNX model using ONNX Runtime
        self.session = ort.InferenceSession(model_path)
        # Get the input and output names for the model
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def predict(self, image):
        # Make prediction using ONNX model
        result = self.session.run(self.output_names, {self.input_name: image})
        return result
    
detmodel = YOLO("assets/retina_detector.pt")
clsmodel = ONNXModel('assets/retina_classifier.onnx')


def get_image(file_path):
    img_path = file_path
    img = Image.open(img_path).convert('RGB').resize((500,500))
    img = np.asarray(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


    
def get_prediction_tags(all_pred):
    prediction_tags = {}
    dr_prediction = all_pred[0][0]
    if dr_prediction[0] >= .25:
        prediction_tags['microaneurysms'] = dr_prediction[0]
    if dr_prediction[1] >= .25:
        prediction_tags['hard_exudates'] = dr_prediction[1]
    if dr_prediction[2] >= .25:
        prediction_tags['cotton_wool_spot'] = dr_prediction[2]
    if dr_prediction[3] >= .25:
        prediction_tags['DME'] = dr_prediction[3]
    if dr_prediction[4] >= .25:
        prediction_tags['venous_looping'] = dr_prediction[4]
    if dr_prediction[5] >= .25:
        prediction_tags['FPD'] = dr_prediction[5]
    if dr_prediction[6] >= .25:
        prediction_tags['FPE'] = dr_prediction[6]
    if dr_prediction[7] >= .25:
        prediction_tags['NVD'] = dr_prediction[7]
    if dr_prediction[8] >= .25:
        prediction_tags['NVE'] = dr_prediction[8]
    if dr_prediction[9] >= .25:
        prediction_tags['viterous_HM'] = dr_prediction[9]
    if dr_prediction[10] >= .25:
        prediction_tags['PreRetinal_HM'] = dr_prediction[10]
    if dr_prediction[11] >= .25:
        prediction_tags['laser'] = dr_prediction[11]
    if all_pred[1][0][0] >= .25:
        prediction_tags['HM'] = all_pred[1][0][0]
    if all_pred[1][0][1] >= .25:
        prediction_tags['VB'] = all_pred[1][0][1]
    if all_pred[1][0][2] >= .25:
        prediction_tags['IRMA'] = all_pred[1][0][2]
    if all_pred[3][0][0] >= .25:
        prediction_tags['no_dr'] = all_pred[3][0][0]
    
    if all_pred[2][0][0] >=.5:
        prediction_tags['ERM'] = all_pred[2][0][0]
    if all_pred[2][0][1] >=.5:
        prediction_tags['Myopic degeneration'] = all_pred[2][0][1]
    if all_pred[2][0][2] >=.5:
        prediction_tags['Drusen'] = all_pred[2][0][2]
    if all_pred[2][0][3] >=.5:
        prediction_tags['Hypertensive Retinopathy'] = all_pred[2][0][3]
    if all_pred[2][0][4] >=.5:
        prediction_tags['Peripapilary Atrophy'] = all_pred[2][0][4]
    if all_pred[2][0][5] >=.5:
        prediction_tags['Tortuous vessels'] = all_pred[2][0][5]
    if all_pred[2][0][6] >=.5:
        prediction_tags['Macular Dystrophy'] = all_pred[2][0][6]
    if all_pred[2][0][7] >=.5:
        prediction_tags['Chorioretinitis'] = all_pred[2][0][7]
    if all_pred[2][0][8] >=.5:
        prediction_tags['Papilledema'] = all_pred[2][0][8]
    if all_pred[2][0][9] >=.5:
        prediction_tags['Nevus'] = all_pred[2][0][9]
    if all_pred[2][0][10] >=.5:
        prediction_tags['FTMH'] = all_pred[2][0][10]
    if all_pred[2][0][11] >=.5:
        prediction_tags['CRVO'] = all_pred[2][0][11]
    if all_pred[2][0][12] >=.5:
        prediction_tags['BRVO'] = all_pred[2][0][12]
    try:
        if all_pred[4][0][0] >=.5:
            prediction_tags['no_disease'] = all_pred[4][0][0]
        if all_pred[3][0][1] >= .25:
            prediction_tags['dr_1'] = all_pred[3][0][1]
        if all_pred[3][0][2] >= .25:
            prediction_tags['dr_2'] = all_pred[3][0][2]
        if all_pred[3][0][2] >= .25:
            prediction_tags['dr_3'] = all_pred[3][0][3]
        if all_pred[3][0][4] >= .25:
            prediction_tags['dr_4'] = all_pred[3][0][4]
        if all_pred[2][0][13] >=.5:
            prediction_tags['Myelination'] = all_pred[2][0][13]
        if all_pred[2][0][14] >=.5:
            prediction_tags['Glaucoma'] = all_pred[2][0][14]
        if all_pred[2][0][15] >=.5:
            prediction_tags['Cataract'] = all_pred[2][0][15]
        if all_pred[2][0][16] >=.5:
            prediction_tags['ME'] = all_pred[2][0][16]
        if all_pred[2][0][17] >=.5:
            prediction_tags['ARMD'] = all_pred[2][0][17]
    except:
        pass
    return prediction_tags
def get_dr_level(pred_dict):
    all_disease = ['microaneurysms', 'hard_exudates', 'cotton_wool_spot', 'DME', 'venous_looping','FPD', 'FPE', 'NVD', 'NVE', 'viterous_HM', 'PreRetinal_HM', 'laser', 'HM', 'VB', 'IRMA','ERM', 'Myopic degeneration', 'Drusen', 'Hypertensive Retinopathy', 'Peripapilary Atrophy', 'Tortuous vessels', 'Macular Dystrophy', 'Chorioretinitis', 'Papilledema', 'Nevus', 'FTMH', 'CRVO', 'BRVO', 'Myelination', 'Glaucoma', 'Cataract', 'ME', 'ARMD', 'dr_1', 'dr_2', 'dr_3', 'dr_4']
    all_keys = ['microaneurysms', 'hard_exudates', 'cotton_wool_spot', 'DME', 'venous_looping','FPD', 'FPE', 'NVD', 'NVE', 'viterous_HM', 'PreRetinal_HM', 'laser', 'HM', 'VB', 'IRMA','no_dr', 'no_disease', 'dr_1', 'dr_2', 'dr_3', 'dr_4']
    dr_4_keys = ['laser','FPD', 'FPE', 'NVD', 'NVE', 'viterous_HM', 'PreRetinal_HM']
    dr_3_keys = ['VB', 'IRMA']
    dr_2_keys = ['hard_exudates', 'cotton_wool_spot', 'venous_looping']
    for key in all_keys:
        try:
            val = pred_dict[key]
        except:
            pred_dict[key] = -1
    for key in all_disease:
        try:
            val = pred_dict[key]
        except:
            pred_dict[key] = -1
    hm_conf = pred_dict['HM']
    dr_2_keys_maxconf = pred_dict[max(dr_2_keys, key=(lambda new_k: pred_dict[new_k]))]
    dr_3_keys_maxconf = pred_dict[max(dr_3_keys, key=(lambda new_k: pred_dict[new_k]))]
    dr_4_keys_maxconf = pred_dict[max(dr_4_keys, key=(lambda new_k: pred_dict[new_k]))]
    disease_maxconf = pred_dict[max(all_disease, key=(lambda new_k: pred_dict[new_k]))]
    if pred_dict['no_disease'] >=.85:
        disease_pred='no_disease'
        grading_pred='dr_0'
        mtm_vtdr = 'non-VTDR'
    else:
        if pred_dict['dr_4'] >.45 or dr_4_keys_maxconf >= .45 or pred_dict['laser'] > .3:
            grading_pred='dr_4'
            mtm_vtdr = 'vtdr'
            disease_pred = 'disease'
        elif pred_dict['dr_3'] >.45 or dr_3_keys_maxconf >= .35 or hm_conf >=.75:
            grading_pred = 'dr_3'
            mtm_vtdr = 'vtdr'
            disease_pred = 'disease' 
        elif pred_dict['dr_2'] >.45 or dr_2_keys_maxconf >= .45 or hm_conf >.4:
            grading_pred = 'dr_2'
            mtm_vtdr = 'non-VTDR'
            disease_pred = 'disease' 
        elif (pred_dict['dr_1'] > pred_dict['no_dr'] and pred_dict['dr_1'] > pred_dict['no_disease']) or (pred_dict['dr_1'] >.45 or pred_dict['microaneurysms'] >= .4):
            grading_pred ='dr_1'
            mtm_vtdr = 'non-VTDR'
            disease_pred = 'disease' 
        elif pred_dict['no_dr'] >.5:
            grading_pred ='dr_0'
            mtm_vtdr = 'non-VTDR'
            if disease_maxconf >=.5: 
                disease_pred = 'disease'
            else:
                disease_pred = 'no_disease'
        else: 
            if pred_dict['no_disease'] >=.5:
                if disease_maxconf >=.5: 
                    disease_pred = 'review'
                else:
                    disease_pred = 'no_disease'
                
                grading_pred ='dr_0'
                mtm_vtdr = 'non-VTDR'
            else:
                disease_pred = 'disease'
                grading_pred ='review'
                mtm_vtdr = 'review'
        
        if min(pred_dict['DME'],pred_dict['hard_exudates'])>=.25 and (pred_dict['DME']+pred_dict['hard_exudates'])>=.8:
            mtm_vtdr = 'vtdr'
    return mtm_vtdr,grading_pred,disease_pred

def cls_predict(img_path, return_dr_level = True):
    prediction_tags = {}
    mtm_vtdr,grading_pred,disease_pred = '', '', ''
    all_pred = clsmodel.predict(get_image(img_path))
    prediction_tags  = get_prediction_tags(all_pred)
    final_pred_tags = prediction_tags.copy()
    if return_dr_level:
        mtm_vtdr,grading_pred,disease_pred = get_dr_level(final_pred_tags)
    return prediction_tags, mtm_vtdr, grading_pred, disease_pred


def transform_classification_result(cls_result_dict):
    transformed_result = {
        "diseases": cls_result_dict[0],                      # Map index 0 to "diseases"
        "vision_threatening": cls_result_dict[1],            # Map index 1 to "vision_threatening"
        "dr_grade": cls_result_dict[2],                      # Map index 2 to "dr_grade"
        "disease_present": cls_result_dict[3]                # Map index 3 to "disease_present"
    }
    return transformed_result
def generate_colors(num_colors):
    np.random.seed(42)  # Seed for reproducibility
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=int)
    return {i: tuple(color) for i, color in enumerate(colors)}

# Define the predict function
def predict(image_path, save=True, conf_threshold=0.25, bounding_box = True):
    # Load the model
    cls_result_dict = cls_predict(image_path)
    cls_result_dict = transform_classification_result(cls_result_dict)
    # Run inference
    results = detmodel.predict(image_path, imgsz=800, show_labels=False, show_conf=False, save=False, conf=conf_threshold)

    # Load the image using OpenCV for drawing
    image = cv2.imread(image_path)

    # Get model class names
    class_names = detmodel.names

    # Generate a unique color for each class
    class_colors = generate_colors(len(class_names))

    # Dictionary to map detected classes to new serial numbers
    detected_classes = {}
    class_serial_mapping = {}  # Mapping from class ID to serial number
    serial_number = 1

    # Result dictionary to hold bounding boxes and confidence scores
    result_dict = {}

    # Iterate over results and process bounding boxes
    for result in results:
        boxes = result.boxes  # Get bounding boxes
        
        for box in boxes:
            # Extract bounding box coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box as integers
            conf = float(box.conf[0])  # Get confidence score
            class_id = int(box.cls)  # Get class id
            
            # Check if the class_id is already mapped to a serial number
            if class_id not in class_serial_mapping:
                class_serial_mapping[class_id] = serial_number
                detected_classes[serial_number] = class_names[class_id]
                serial_number += 1

            # Get the new serial number for this class
            class_number = class_serial_mapping[class_id]
            if bounding_box:
                # Add the bounding box and confidence to the result dictionary
                result_dict[class_names[class_id]] = {
                    "bounding_box": [x1, y1, x2, y2],
                    "confidence": conf
                }
            else:
                result_dict[class_names[class_id]] = {
                    "bounding_box": [x1, y1, x2, y2],
                    "confidence": conf
                }

            if save:
                # Get color for this class and ensure it is a tuple
                color = class_colors[class_id]
                # Debugging: Check color value

                # Verify the color is in the correct format
                if isinstance(color, tuple) and len(color) == 3:
                    # Ensure each value is an integer
                    color = tuple(map(int, color))
                else:
                    # Default to white if color is not valid
                    color = (255, 255, 255)

                # Draw bounding box rectangle with a slightly thicker border for visibility
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw the serial number instead of class name with a filled rectangle background
                text = str(class_number)
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
                cv2.putText(image, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if save:
        # Create a legend background with padding and a slight border
        legend_height = 40 * len(detected_classes) + 20
        legend_width = image.shape[1]
        legend_background = np.full((legend_height, legend_width, 3), (255, 255, 255), dtype=np.uint8)
        cv2.rectangle(legend_background, (0, 0), (legend_width, legend_height), (0, 0, 0), 2)

        # Draw the legend with a filled background for each class
        for i, (serial, class_name) in enumerate(detected_classes.items()):
            text = f"{serial}. {class_name}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            legend_x, legend_y = 10, 40 * (i + 1)
            color = class_colors[list(class_serial_mapping.keys())[list(class_serial_mapping.values()).index(serial)]]
            # Debugging: Check legend color

            if isinstance(color, tuple) and len(color) == 3:
                # Ensure each value is an integer
                color = tuple(map(int, color))
            else:
                color = (255, 255, 255)

            cv2.rectangle(legend_background, (legend_x - 5, legend_y - text_height - 5), (legend_x + text_width + 5, legend_y + baseline), color, -1)
            cv2.putText(legend_background, text, (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Concatenate the legend with the original image
        output_image = np.concatenate((image, legend_background), axis=0)

        # Save the results image to a temporary file
        result_uuid = str(uuid.uuid4())
        output_image_path = f".output_with_bboxes_and_legend{result_uuid}.jpg"

        cv2.imwrite(output_image_path, output_image)
    else:
        output_image_path = None

    return output_image_path, result_dict, cls_result_dict



# Full-width layout
# st.set_page_config(page_icon="logo_small.png", layout="wide", page_title="Fundus App")

# Custom CSS to add padding at the top and style the slider
st.markdown("""
    <style>
    .main-content {
        padding-top: 50px;  /* Adjust this value to control the starting position of the page */
    }
    .stSlider {
        margin-bottom: 25px;  /* Add some space below the slider */
    }
    .stSlider > div {
        height: 60px;  /* Adjust the height of the slider */
        padding: 10px;  /* Add padding around the slider */
        background-color: #f4f4f4;  /* Light background for the slider */
        border: 1px solid #ccc;  /* Light border */
        border-radius: 10px;  /* Rounded corners */
    }
    </style>
""", unsafe_allow_html=True)


_, col1, _ = st.columns([1,3,1])
with col1:
    st.write("Upload a Fundus image to get diagnosis")
    # File uploader for custom image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tiff"], help="Upload your retinal image here.")
with col1:
    # Confidence threshold slider with improved style
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, help="Adjust the threshold for object detection confidence.")

# Specify the default folder ID for image upload
config = load_config()
default_folder_id = config["google_drive"]["folders"]["default"]

# Initialize Google Drive service once
service = create_drive_service()

# Initialize session state for detection completion and feedback
if 'detection_done' not in st.session_state:
    st.session_state.detection_done = False

if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False


button_style = """
    <style>
    .stButton > button {
        background-color: #4CAF50;  /* Green background */
        color: white;  /* White text */
        border: none;  /* Remove border */
        padding: 12px 24px;  /* Padding for the button */
        text-align: center;  /* Centered text */
        text-decoration: none;  /* Remove underline */
        display: inline-block;  /* Inline block */
        font-size: 16px;  /* Text size */
        margin: 4px 2px;  /* Space between buttons */
        transition-duration: 0.4s;  /* Animation */
        cursor: pointer;  /* Pointer cursor on hover */
        border-radius: 8px; /* Rounded corners */
    }

    .stButton > button:hover {
        background-color: white; 
        color: black; 
        border: 2px solid #4CAF50; /* Green border on hover */
    }
    </style>
"""

st.markdown(button_style, unsafe_allow_html=True)

# Check if a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file temporarily
    img = Image.open(uploaded_file)
    temp_dir = tempfile.mkdtemp()
    # Generate a unique filename using UUID
    unique_filename = f"uploaded_image_{uuid.uuid4().hex}.jpg"
    image_path = f"{temp_dir}/{unique_filename}"
    img.save(image_path)

    # Upload the file to the specified folder in Google Drive
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
        file_id = upload_file_to_drive(service, uploaded_file, default_folder_id)
        
    _, col1, _ = st.columns([1,3,1])
    with col1:
        # Detect disease button
        if st.button("Run Prediction", type="primary", use_container_width=True):
            # Perform prediction (Assuming `predict` function is defined elsewhere)
            saved_path, detection_details, classification_details = predict(image_path, save=True, conf_threshold=conf_threshold)
        
            # Create columns for side-by-side layout with custom width ratios
            col1, col2 = st.columns([2, 1])  # 2/3 width for image, 1/3 width for details

            # Display the saved image with bounding boxes in the left column
            with col1:
                if saved_path:
                    st.image(saved_path, caption="Detected Objects with Bounding Boxes", use_column_width=True)

            # Display detection and classification details in the right column
            with col2:
                # Transform Detected Diseases
                diseases = classification_details.get('diseases', {})
                vision_threatening = classification_details.get('vision_threatening', 'Absent')
                dr_grade = classification_details.get('dr_grade', '')
                disease_present = classification_details.get('disease_present', 'Absent')

                # Convert disease keys to user-friendly names
                disease_map = {
                    'HM': 'Hemorrhage',
                    'dr_1': 'DR (Stage 1)',
                    'dr_2': 'DR (Stage 2)',
                    'dr_3': 'DR (Stage 3)',
                    'dr_4': 'DR (Stage 4)',
                    'hard_exudates': 'Hard Exudates',
                    'cotton_wool_spot': 'Soft Exudates',
                    'venous_looping': 'Venous Looping',
                    'viterous_HM': 'Viterous Hemorrhage',
                    'PreRetinal_HM': 'Preretinal Hemorrhage',
                    'VB': 'Venous Beading',
                    'IRMA': 'Intraretinal Microvascular Abnormality',
                    'FPD' : 'Fibrous-Proliferation (Disk)',
                    'FPE' : 'Fibrous-Proliferation (Elsewhere)',
                    'NVD' : 'Neovascularization (Disk)',
                    'NVE': 'Neovascularization (Elsewhere)',
                    'FTMH': 'Full Thickness Macular Hole',
                    'ME': 'Macular Edema',
                    'ERM': 'Epiretinal Membrane',
                    'laser': 'Laser',
                    'microaneurysms': 'Microaneurysms',
                    'no_dr': 'No DR',
                    'no_disease': 'No Disease'
                }
                transformed_diseases = [disease_map.get(d, d) for d in diseases.keys()]

                # Retina Status
                st.markdown("**Eye Status**")
                st.write(f"{disease_present.capitalize()}")

                # Vision Threatening
                st.markdown("**Vision Threatening**")
                vision_status = 'Yes' if vision_threatening == 'vtdr' else 'No'
                st.write(f"{vision_status.capitalize()}")

                # Custom container for diseases
                st.markdown("""<style>
                    .custom-container {
                        background-color: #f9f9f9;
                        padding: 10px;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        max-height: 150px;
                        overflow-y: auto;
                    }
                    </style>""", unsafe_allow_html=True)

                st.markdown("**Diseases Present**")
                with st.container():
                    diseases_text = '<br>'.join(transformed_diseases) if transformed_diseases else 'None'
                    st.markdown(f'<div class="custom-container">{diseases_text}</div>', unsafe_allow_html=True)

                # Expandable text boxes for detailed JSON view
                with st.expander("Classification JSON"):
                    st.json(classification_details)
                with st.expander("Detection JSON"):
                    st.json(detection_details)

            # Mark detection as done in session state
            st.session_state.detection_done = True
            st.session_state.feedback_given = False  # Reset feedback state for next detection



##FEEDBACK SYSTEM 
# if st.session_state.detection_done:

#     show_sidebar_style = """
#         <style>
#         [data-testid="stSidebar"] {display: block;}
#         </style>
#         """
#     st.markdown(show_sidebar_style, unsafe_allow_html=True)
#     # Sidebar content for feedback
#     with st.sidebar:
#         # Apply custom styles for spacing between components
#         st.markdown("""<style>
#         .app-spacing {
#             margin-top: -15px;
#             margin-bottom: -30px;
#         }
#         .button-spacing {
#             margin-bottom: 10px;
#         }
#         </style>""", unsafe_allow_html=True)

#         # Improved CSS for the classification message with cleared margins
#         classification_style = """
#             <style>
#             /* Container styling with no margin at top and bottom */
#             .classification-container {
#                 padding: 11px;
#                 background-color: #f7f7f7;
#                 border: 1px solid #ddd;
#                 border-radius: 10px;
#                 box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.1);
#                 text-align: center;
#                 font-family: 'Arial', sans-serif;
#                 margin-top: -40px;  /* Remove top margin */
#                 margin-bottom: -35px;  /* Remove bottom margin */
#             }

#             /* Styling for the heading */
#             .classification-container h1 {
#                 color: #22686E;
#                 font-size: 13px;
#                 font-weight: bold;
#                 margin: 0;  /* Remove margins around the heading */
#                 font-family: 'Arial', sans-serif;
#             }
#             </style>
#         """

#         # Apply the custom style
#         st.markdown(classification_style, unsafe_allow_html=True)

#         # Display the classification message inside a styled container
#         classification_message = """
#         <div class='classification-container'>
#             <h1>Your feedback improves predictions for more accurate results. Was this prediction correct?</h1>
#         </div>
#         """
#         st.markdown(classification_message, unsafe_allow_html=True)

#         # Insert a space after the markdown to ensure proper display order
#         st.markdown("<div class='button-spacing'></div>", unsafe_allow_html=True)

#         # Placeholder for the success message
#         message_placeholder = st.empty()

#         # Display feedback buttons in the sidebar only if feedback hasn't been given
#         if not st.session_state.feedback_given:
#             # Define folder IDs for feedback
#             correct_pred_folder_id = "1ncxH-PxJW7nJu9Rlwm4LUN3DQg8BGh2f"
#             incorrect_pred_folder_id = "1oUSNdHGv_jiqSfFi92kniWCnONxlnbR7"
#             partially_pred_folder_id = "1F8x6zMTuReS2_BkUMlJNZcLGbWhyw6Nv"

#             if st.button("Correct", use_container_width=True):
#                 upload_file_to_drive(service, uploaded_file, correct_pred_folder_id)
#                 st.session_state.feedback_given = True

#             if st.button("Incorrect", use_container_width=True):
#                 upload_file_to_drive(service, uploaded_file, incorrect_pred_folder_id)
#                 st.session_state.feedback_given = True

#             if st.button("Partially correct", use_container_width=True):
#                 upload_file_to_drive(service, uploaded_file, partially_pred_folder_id)
#                 st.session_state.feedback_given = True

#         # Hide all buttons once feedback is given
#         if st.session_state.feedback_given:
#             st.session_state.detection_done = False  # Mark detection as not done to hide buttons
#             message_placeholder.success("Thank you for your feedback!")
#             # Pause for 1 seconds
#             time.sleep(1)
#             # Clear the success message
#             message_placeholder.empty()
