import os
import numpy as np
import cv2
import warnings
import logging
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from datetime import datetime
from flask import send_file

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Disable GPU for stable inference
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from treatment import get_treatment_recommendation
from predictions import predict_skin_disease,resnet_model
from gradcam import generate_resnet50_gradcam

class_names = {
    0: "Actinic Keratosis",
    1: "Dermatofibroma",
    2: "Nevus",
    3: "Pigmented Benign Keratosis",
    4: "Seborrheic Keratosis",
    5: "Vascular Lesion"
}

def validate_image(img_path):

    img = cv2.imread(img_path)

    if img is None:
        return False, "Invalid image file."

    if np.std(img) < 15:
        return False, "Image quality too low."

    return True, ""

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():

    print("🔥 ENTERED /detect")

    try:
        if request.method == 'POST':
            print("📩 POST request received")

            if 'file' not in request.files:
                print("❌ No file in request")
                return "No file uploaded"

            file = request.files['file']
            print("📂 File received:", file.filename)

            if file.filename == '':
                print("❌ Empty filename")
                return "No file selected"

            filename = secure_filename(file.filename)

            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            print("✅ File saved:", img_path)

            # ---------------- VALIDATION ----------------
            print("🔍 Validating image...")
            valid, message = validate_image(img_path)

            if not valid:
                print("❌ Validation failed:", message)
                return render_template(
                    'result.html',
                    uploaded_image=url_for('static', filename='uploads/' + filename),
                    gradcam_image=None,
                    disease_name="Invalid Image",
                    confidence=0,
                    treatment=message
                )

            print("✅ Validation passed")

            # ---------------- PREDICTION ----------------
            print("🧠 Running prediction...")
            pred_class, confidence = predict_skin_disease(img_path)
            print(f"📊 Prediction done: class={pred_class}, conf={confidence}")

            if confidence < 60:
                print("⚠️ Low confidence")
                return render_template(
                    'result.html',
                    uploaded_image=url_for('static', filename='uploads/' + filename),
                    gradcam_image=None,
                    disease_name="Unknown / Not a skin image",
                    confidence=confidence,
                    treatment="Please upload a clear skin lesion image."
                )

            disease_name = class_names.get(pred_class, "Unknown")
            print("🧾 Disease:", disease_name)

            treatment_details = get_treatment_recommendation(disease_name)
            print("💊 Treatment fetched")

            # ---------------- GRADCAM ----------------
            print("🔥 Generating GradCAM...")
            gradcam_output = os.path.join("static/uploads", "gradcam_" + filename)

            generate_resnet50_gradcam(img_path, resnet_model, gradcam_output)

            if not os.path.exists(gradcam_output):
                print("❌ GradCAM not generated!")
                gradcam_url = None
            else:
                print("✅ GradCAM saved:", gradcam_output)
                gradcam_url = url_for('static', filename='uploads/' + "gradcam_" + filename)

            # ---------------- REPORT DATA ----------------
            global latest_report_data
            latest_report_data = {
                "disease": disease_name,
                "confidence": confidence,
                "description": treatment_details['description'],
                "treatment": treatment_details['medical_treatment'],
                "doctor_advice": treatment_details['doctor_advice'],
                "uploaded_image": img_path,
                "gradcam_image": gradcam_output
            }

            print("🎯 Rendering result page...")

            return render_template(
                'result.html',
                uploaded_image=url_for('static', filename='uploads/' + filename),
                gradcam_image=gradcam_url,
                disease_name=disease_name,
                confidence=confidence,
                treatment=treatment_details
            )

        print("📄 GET request → loading detect page")
        return render_template('detect.html')

    except Exception as e:
        print("🚨 ERROR OCCURRED:", str(e))
        return f"Internal Error: {str(e)}"


from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from datetime import datetime
import os


def generate_medical_report(data):

    pdf_path = f"static/reports/medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    os.makedirs("static/reports", exist_ok=True)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle(
        'title',
        parent=styles['Title'],
        alignment=1,
        textColor=colors.HexColor("#004aad"),
        fontSize=18,
        spaceAfter=10
    )

    header_style = ParagraphStyle(
        'header',
        parent=styles['Heading2'],
        textColor=colors.HexColor("#004aad"),
        spaceAfter=6
    )

    normal = ParagraphStyle(
        'normal',
        parent=styles['Normal'],
        spaceAfter=6,
        leading=14
    )

    disclaimer_style = ParagraphStyle(
        'disc',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )

    elements.append(Paragraph("AI Dermatology Diagnostic Report", title_style))

    elements.append(Paragraph(
        f"<b>Generated on:</b> {datetime.now().strftime('%d %B %Y  %H:%M')}",
        normal
    ))

    elements.append(Spacer(1, 10))
    severity = "Low Risk"
    if data['confidence'] > 90:
        severity = "Moderate Risk"
    if data['confidence'] > 97:
        severity = "High Risk"
    info_table = Table([
        ["Disease Detected", data['disease']],
        ["Confidence Level", f"{data['confidence']} %"],
        ["AI Risk Level", severity]
    ], colWidths=[160, 340])


    info_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
    ('BOX', (0,0), (-1,-1), 1.2, colors.grey),
    ('INNERGRID', (0,0), (-1,-1), 0.6, colors.grey),
    ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
    ('PADDING', (0,0), (-1,-1), 6)]))

    elements.append(info_table)

    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Clinical Description", header_style))
    elements.append(Paragraph(data['description'], normal))

    elements.append(Paragraph("Recommended Medical Treatment", header_style))

    for item in data['treatment']:
        elements.append(Paragraph(f"• {item}", normal))

    elements.append(Paragraph("Doctor Guidance", header_style))
    elements.append(Paragraph(data['doctor_advice'], normal))

    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Diagnostic Images", header_style))

    uploaded_title = Paragraph(
    "<para align='center'><b>Original Skin Image</b></para>", normal)
    gradcam_title = Paragraph(
    "<para align='center'><b>AI GradCAM Heatmap</b></para>", normal)


    uploaded_img = Image(data['uploaded_image'], width=240, height=190)
    gradcam_img = Image(data['gradcam_image'], width=240, height=190)

    image_table = Table([
        [uploaded_title, gradcam_title],
        [uploaded_img, gradcam_img]
    ], colWidths=[250, 250])

    image_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1.2, colors.HexColor("#004aad")),
        ('INNERGRID', (0,0), (-1,-1), 0.8, colors.grey),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('ALIGN', (0,1), (-1,1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 8)
    ]))

    elements.append(image_table)

    elements.append(Spacer(1, 12))

    elements.append(Paragraph("AI Medical Disclaimer", header_style))

    disclaimer_text ="""
This AI-generated report is for clinical decision support only and should not be
used for self-diagnosis or treatment. A licensed dermatologist must review
the findings before medical decisions are made.
"""


    elements.append(Paragraph(disclaimer_text, disclaimer_style))

    elements.append(Spacer(1, 10))

    elements.append(Paragraph(
        "© AI Dermatology System — Clinical Intelligence Report",
        disclaimer_style
    ))

    doc.build(elements)

    return pdf_path
@app.route('/download-report')
def download_report():

    global latest_report_data

    if not latest_report_data:
        return "Run detection first before downloading report."

    pdf_path = generate_medical_report(latest_report_data)

    return send_file(pdf_path, as_attachment=True)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/features')
def features():
    return render_template('features.html')


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)