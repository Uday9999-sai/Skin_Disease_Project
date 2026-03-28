# ==========================================================
# ADVANCED MEDICAL TREATMENT & CLINICAL REPORT ENGINE
# ==========================================================

def get_treatment_recommendation(disease_name):

    treatments = {

        "Actinic Keratosis": {
            "description": "A precancerous skin lesion caused by long-term sun exposure. Early treatment is important to prevent progression into squamous cell carcinoma.",
            
            "symptoms": [
                "Rough scaly patches",
                "Dry or crusted skin lesion",
                "Burning or itching sensation"
            ],

            "causes": [
                "Chronic UV radiation exposure",
                "Aging skin",
                "Outdoor occupational exposure"
            ],

            "medical_treatment": [
                "Cryotherapy (freezing abnormal cells)",
                "Topical chemotherapy creams (5-FU, imiquimod)",
                "Photodynamic therapy",
                "Laser therapy for resistant lesions"
            ],

            "self_care": [
                "Apply SPF 50+ sunscreen daily",
                "Avoid peak sun hours",
                "Wear protective clothing and hats"
            ],

            "prevention": [
                "Routine skin checks",
                "Avoid tanning beds",
                "Use broad-spectrum sunscreen"
            ],

            "risk_level": "Moderate (precancerous)",

            "doctor_advice": "Immediate dermatology consultation recommended to prevent skin cancer transformation."
        },

        "Dermatofibroma": {
            "description": "A benign fibrous skin tumor usually harmless and slow-growing.",
            
            "symptoms": [
                "Firm small bump",
                "Brown or reddish lesion",
                "Tenderness when pressed"
            ],

            "causes": [
                "Minor skin injury",
                "Insect bites",
                "Folliculitis"
            ],

            "medical_treatment": [
                "No treatment required in most cases",
                "Surgical excision if painful",
                "Laser removal for cosmetic reasons"
            ],

            "self_care": [
                "Avoid scratching",
                "Monitor size or color change"
            ],

            "prevention": [
                "Skin hygiene",
                "Avoid trauma to skin"
            ],

            "risk_level": "Low",

            "doctor_advice": "Consult if lesion becomes painful, grows rapidly, or changes color."
        },

        "Nevus": {
            "description": "A mole (nevus) is typically benign but must be monitored for melanoma risk.",
            
            "symptoms": [
                "Dark pigmented patch",
                "Flat or raised mole",
                "Uniform or irregular borders"
            ],

            "causes": [
                "Genetic predisposition",
                "Sun exposure",
                "Melanocyte clustering"
            ],

            "medical_treatment": [
                "Dermoscopic examination",
                "Biopsy if suspicious",
                "Surgical removal if malignant risk"
            ],

            "self_care": [
                "Monitor ABCDE changes",
                "Avoid UV radiation",
                "Use sunscreen"
            ],

            "prevention": [
                "Annual skin check",
                "Avoid tanning",
                "Early detection monitoring"
            ],

            "risk_level": "Variable (depends on changes)",

            "doctor_advice": "Visit dermatologist if mole changes in size, color, or shape."
        },

        "Pigmented Benign Keratosis": {
            "description": "A non-cancerous pigmented lesion common in aging skin.",
            
            "symptoms": [
                "Dark waxy patch",
                "Slightly raised surface",
                "Mild irritation sometimes"
            ],

            "causes": [
                "Age-related skin changes",
                "Sun exposure",
                "Genetic factors"
            ],

            "medical_treatment": [
                "Cryotherapy",
                "Laser removal",
                "Topical soothing creams"
            ],

            "self_care": [
                "Keep skin moisturized",
                "Avoid harsh chemicals"
            ],

            "prevention": [
                "Routine skin care",
                "Sun protection"
            ],

            "risk_level": "Low",

            "doctor_advice": "Consult if itching, bleeding, or sudden enlargement occurs."
        },

        "Seborrheic Keratosis": {
            "description": "A very common benign growth seen in adults and elderly.",
            
            "symptoms": [
                "Brown/black growth",
                "Waxy appearance",
                "Raised lesion"
            ],

            "causes": [
                "Aging",
                "Genetic predisposition",
                "Sun exposure"
            ],

            "medical_treatment": [
                "Cryotherapy",
                "Curettage",
                "Laser therapy"
            ],

            "self_care": [
                "Avoid picking",
                "Maintain skin hygiene"
            ],

            "prevention": [
                "Skin monitoring",
                "Sun protection"
            ],

            "risk_level": "Low",

            "doctor_advice": "Seek consultation if lesion changes rapidly."
        },

        "Vascular Lesion": {
            "description": "Lesions formed due to abnormal blood vessel growth or dilation.",
            
            "symptoms": [
                "Red or purple patch",
                "Blanching on pressure",
                "Occasional bleeding"
            ],

            "causes": [
                "Congenital factors",
                "Trauma",
                "Circulatory abnormalities"
            ],

            "medical_treatment": [
                "Laser therapy",
                "Electrosurgery",
                "Medication in severe cases"
            ],

            "self_care": [
                "Avoid trauma",
                "Protect area"
            ],

            "prevention": [
                "Regular monitoring"
            ],

            "risk_level": "Moderate",

            "doctor_advice": "Consult specialist for treatment planning."
        },

        "Unknown": {
            "description": "AI could not determine a confident diagnosis.",
            
            "symptoms": [],
            "causes": [],
            "medical_treatment": [
                "Immediate dermatology consultation required"
            ],
            "self_care": [
                "Avoid self-medication"
            ],
            "prevention": [],
            "risk_level": "Unknown",

            "doctor_advice": "Clinical examination and dermoscopy required."
        }
    }

    return treatments.get(disease_name, treatments["Unknown"])
