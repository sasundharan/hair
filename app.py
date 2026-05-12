from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Hairstyle Recommendations with External Image URLs
men_face_shapes = {
    "Oval": [
                {"name": "Slicked Back", "img": "https://i.pinimg.com/736x/c1/3c/d4/c13cd4b116dcb71e324e04ddae3ceaf5.jpg", "procedure": "1. Grow your hair to at least 3-4 inches on top.\n2. Trim the sides and back short or taper them for a clean look.\n3. Wash and towel-dry your hair to prepare for styling.\n4. Apply a styling gel or pomade to damp hair.\n5. Comb your hair back smoothly using a fine-tooth comb.\n6. Blow-dry your hair in the direction of the slick back for hold.\n7. Add more product if needed for a sleek, polished finish.\n8. Smooth any flyaways and ensure a clean look.\n9. Finish with hairspray to lock the style in place.\n10. Maintain the style with regular trims and styling."},
		 {"name": "Undercut", "img": "https://i.pinimg.com/originals/3a/0d/38/3a0d38349a8393ec74d2f7dcf4ddbc5e.jpg", "procedure": "1. Decide on the length for the top and the undercut section.\n2. Section off the top hair and clip it out of the way.\n3. Use clippers to shave the sides and back to your desired length.\n4. Blend the transition between the top and sides for a seamless look.\n5. Trim the top hair to your preferred length and style.\n6. Clean up the edges around the ears and neckline.\n7. Wash and dry your hair to remove loose strands.\n8. Style the top with pomade, wax, or gel for a polished look.\n9. Maintain the undercut by trimming the sides every 2-3 weeks.\n10. Experiment with styling the top for different looks."},

                {"name": "Quiff", "img": "https://i.pinimg.com/736x/b6/77/82/b67782e2887c3cdd90b4d704324e2b42.jpg", "procedure": "1. Grow your hair to at least 3-5 inches on top for volume.\n2. Trim the sides and back short or taper them for contrast.\n3. Wash and towel-dry your hair to prepare for styling.\n4. Apply a volumizing mousse or pre-styler to damp hair.\n5. Blow-dry the top hair upward and forward using a round brush.\n6. Use a strong-hold pomade or wax to shape the quiff.\n7. Comb the hair upward and slightly back to create height.\n8. Smooth the sides and ensure a clean, polished look.\n9. Finish with hairspray to hold the quiff in place.\n10. Maintain the style with regular trims and styling."}
            ],
            "Square": [
                {"name": "Buzz Cut", "img": "https://i.pinimg.com/736x/e2/68/78/e26878c1b2735652021c387641e1cadd.jpg", "procedure": "1. Choose the clipper guard size based on your desired hair length.\n2. Start with clean, dry hair for an even cut.\n3. Begin at the sides and back, moving the clippers upward in smooth motions.\n4. Move to the top and cut in the direction of hair growth.\n5. Check for evenness and go over any uneven spots.\n6. Clean up the hairline around the ears and neck with a trimmer.\n7. Rinse off loose hair and pat your scalp dry.\n8. Apply a moisturizer or aftershave to soothe the skin.\n9. Maintain the cut by trimming every 2-3 weeks.\n10. Protect your scalp with sunscreen when outdoors."},
                {"name": "Side Part", "img": "https://i.pinimg.com/736x/72/fc/10/72fc101c8ad808a1f4271ec94b2062d4.jpg", "procedure": "1. Grow your hair to at least 2-4 inches on top for flexibility.\n2. Trim the sides and back short or taper them for a clean look.\n3. Wash and towel-dry your hair to prepare for styling.\n4. Apply a styling product like pomade, wax, or gel to damp hair.\n5. Use a comb to create a clean part on your preferred side.\n6. Comb the hair on one side smoothly toward the part.\n7. Style the other side with a slight curve or wave for volume.\n8. Smooth any flyaways and ensure a polished finish.\n9. Finish with hairspray to hold the style in place.\n10. Maintain the look with regular trims and styling."}
            ],
            "Round": [
                {"name": "Textured Pompadour", "img": "https://i.pinimg.com/736x/05/48/e6/0548e60ca77d5d376ab871df86aca1ad.jpg", "procedure": "1. Grow your hair to at least 4-6 inches on top for volume and texture.\n2. Trim the sides and back short or taper them for contrast.\n3. Wash and towel-dry your hair to prepare for styling.\n4. Apply a volumizing mousse or pre-styler to damp hair.\n5. Blow-dry the top hair upward and backward using a round brush.\n6. Use a texturizing paste or wax to create separation and definition.\n7. Comb the hair upward and back, then tousle for a textured look.\n8. Smooth the sides and ensure a clean, polished finish.\n9. Finish with hairspray to hold the textured pompadour in place.\n10. Maintain the style with regular trims and styling."},
                {"name": "Spiky Hair", "img": "https://i.pinimg.com/736x/3d/c9/a5/3dc9a57a0e3f682bbc04524ce038e911.jpg", "procedure": "1. Grow your hair to at least 2-4 inches for spikes to stand out.\n2. Trim the sides and back short or taper them for contrast.\n3. Wash and towel-dry your hair to prepare for styling.\n4. Apply a strong-hold gel, wax, or pomade to damp or dry hair.\n5. Use your fingers to pull small sections of hair upward.\n6. Twist and pinch the ends to create sharp spikes.\n7. Adjust the spikes for height and direction as desired.\n8. Smooth the sides and ensure a clean, polished look.\n9. Finish with hairspray to hold the spikes in place.\n10. Maintain the style with regular trims and styling."}
            ],
            "Heart": [
                {"name": "Long Fringe", "img": "https://i.pinimg.com/736x/c4/62/1b/c4621b7496d0745b3eb3c42425d4cc5b.jpg", "procedure": "1. Grow your hair long enough to cover your forehead or eyes.\n2. Trim the sides and back to your desired length for balance.\n3. Wash and towel-dry your hair to prepare for styling.\n4. Apply a lightweight styling product like mousse or texturizing spray.\n5. Blow-dry the fringe forward or to the side for volume and shape.\n6. Use a comb or fingers to style the fringe into place.\n7. Add texture by lightly tousling the fringe for a casual look.\n8. Smooth any flyaways and ensure a polished finish.\n9. Finish with light-hold hairspray to keep the fringe in place.\n10. Maintain the style with regular trims to avoid overgrowth."},
                {"name": "Side Swept", "img": "https://i.pinimg.com/736x/03/af/f1/03aff18311d77d26636a899f6d15bf39.jpg", "procedure": "1. Grow your hair to at least 3-5 inches on top for volume.\n2. Trim the sides and back short or taper them for contrast.\n3. Wash and towel-dry your hair to prepare for styling.\n4. Apply a styling product like pomade, wax, or mousse to damp hair.\n5. Use a comb to create a side part on your preferred side.\n6. Blow-dry the top hair toward the side for a swept-back look.\n7. Comb the hair smoothly to the side for a polished finish.\n8. Add texture by lightly tousling the top for a casual vibe.\n9. Finish with hairspray to hold the side-swept style in place.\n10. Maintain the look with regular trims and styling."}
            ],
            "Diamond": [
                {"name": "Textured Crop", "img": "https://i.pinimg.com/736x/e9/1a/96/e91a968086385a2ae26ffe867ea6760e.jpg", "procedure": "1. Grow your hair to at least 2-3 inches on top for texture.\n2. Trim the sides and back short or fade them for contrast.\n3. Wash and towel-dry your hair to prepare for styling.\n4. Apply a texturizing product like paste, wax, or salt spray.\n5. Use your fingers to tousle the top hair for a messy, textured look.\n6. Blow-dry the hair upward and forward for added volume.\n7. Define the texture by pinching and separating small sections of hair.\n8. Smooth the sides and ensure a clean, polished finish.\n9. Finish with light-hold hairspray to hold the texture in place.\n10. Maintain the style with regular trims and styling."},
                {"name": "Faux Hawk", "img": "https://i.pinimg.com/736x/93/05/d9/9305d96e2d6686b0a986931bf6d2b3f0.jpg", "procedure": "1. Grow your hair to at least 3-5 inches on top for height.\n2. Trim the sides and back short or fade them for contrast.\n3. Wash and towel-dry your hair to prepare for styling.\n4. Apply a strong-hold product like gel, wax, or pomade to damp hair.\n5. Use your fingers or a comb to pull the top hair upward into a ridge.\n6. Shape the sides to stay flat while keeping the center raised.\n7. Define the faux hawk by smoothing and adjusting the height.\n8. Add texture by lightly tousling the top for a rugged look.\n9. Finish with hairspray to hold the faux hawk in place.\n10. Maintain the style with regular trims and styling."}
            ],
            "Oblong": [
               {"name": "Long and Wavy", "img": "https://i.pinimg.com/736x/e2/86/cb/e286cb1e41bd052a80ab509819a230b6.jpg", "procedure": "1. Grow your hair to at least shoulder length for the desired wavy look.\n2. Wash and condition 2-3 times a week using sulfate-free products to keep hair healthy.\n3. Towel dry gently to avoid frizz and damage.\n4. Apply styling products like heat protectant and curl-enhancing cream.\n5. Create waves by air-drying, using a diffuser, or heat tools like a curling wand.\n6. Add texture with sea salt spray for a natural, beachy look.\n7. Style and shape waves using your fingers or a wide-tooth comb.\n8. Set the style with light-hold hairspray for long-lasting waves.\n9. Maintain overnight by sleeping on a silk pillowcase or tying hair loosely.\n10. Refresh waves with water or leave-in conditioner and trim regularly to prevent split ends."},
                {"name": "Layered Hair", "img": "https://i.pinimg.com/736x/4e/e1/8c/4ee18cf5bf5671d13ca62a9cf033e238.jpg", "procedure": "1. Consult a stylist to discuss your desired layered look and bring reference photos.\n2. Choose the type of layers (light, heavy, face-framing, or uniform) based on your hair type and style.\n3. Wash and condition your hair with sulfate-free products to keep it healthy.\n4. Section and cut your hair at different lengths to create layers.\n5. Style with lightweight products like mousse or texturizing spray to enhance layers.\n6. Blow-dry with a round brush to add volume and shape to the layers.\n7. Finish with a light-hold hairspray or wax to define the layers.\n8. Maintain with regular trims every 4-6 weeks to keep the shape.\n9. Protect from heat damage by using a heat protectant when styling.\n10. Experiment with styling to find the best look for your layered haircut."}
            ],
}

women_face_shapes = {
    "Oval": [
        {"name": "Long Layers", "img": "https://i.pinimg.com/736x/e8/a2/d8/e8a2d8b67bd0ae270502a65ddb2baaaa.jpg", "procedure": "1. Grow your hair to at least shoulder length or longer.\n2. Consult a stylist to add layers for movement and volume.\n3. Wash and condition with sulfate-free products for healthy hair.\n4. Towel-dry gently and apply a heat protectant.\n5. Blow-dry your hair using a round brush to enhance layers.\n6. Use a curling iron or flat iron to add soft waves.\n7. Apply a texturizing spray to define the layers.\n8. Comb through gently to separate and shape the layers.\n9. Finish with hairspray to hold the style.\n10. Maintain with regular trims every 6-8 weeks."},
        {"name": "Bob", "img": "https://i.pinimg.com/736x/59/31/19/5931196ac6b526dd8c25309f2bf7fdff.jpg", "procedure": "1. Choose your bob length (chin-length, shoulder-length, or angled).\n2. Consult a stylist for a precise cut.\n3. Wash and condition with hydrating products.\n4. Towel-dry gently and apply a heat protectant.\n5. Blow-dry your hair straight or with a slight inward curve.\n6. Use a flat iron for a sleek, polished look.\n7. Add texture with a texturizing spray or wax.\n8. Style the ends to flip inward or outward for variation.\n9. Finish with hairspray for hold.\n10. Maintain with trims every 4-6 weeks."}
    ],
    "Square": [
        {"name": "Side Bangs", "img": "https://i.pinimg.com/736x/4a/4a/4d/4a4a4d4e4c8f4e6d4b4e4e0d4c4a4e4d.jpg", "procedure": "1. Decide on the length of your side bangs (eyebrow, cheekbone, or chin-length).\n2. Consult a stylist to cut the bangs at an angle.\n3. Wash and condition your hair as usual.\n4. Towel-dry gently and apply a heat protectant.\n5. Blow-dry the bangs to the side using a round brush.\n6. Use a flat iron to smooth or curl the bangs slightly.\n7. Style the rest of your hair as desired.\n8. Apply a light-hold spray to keep the bangs in place.\n9. Trim the bangs every 2-3 weeks to maintain length.\n10. Experiment with styling for different looks."},
        {"name": "Wavy Lob", "img": "https://i.pinimg.com/736x/5b/4a/4d/5b4a4d4e4c8f4e6d4b4e4e0d4c4a4e4d.jpg", "procedure": "1. Grow your hair to a lob (long bob) length (shoulder-length).\n2. Consult a stylist to add soft layers for movement.\n3. Wash and condition with hydrating products.\n4. Towel-dry gently and apply a heat protectant.\n5. Blow-dry your hair with a round brush for volume.\n6. Use a curling wand to create loose waves.\n7. Apply a texturizing spray for a beachy look.\n8. Scrunch the waves with your hands for added texture.\n9. Finish with hairspray to hold the waves.\n10. Maintain with trims every 6-8 weeks."}
    ],
    "Round": [
        {"name": "High Ponytail", "img": "https://i.pinimg.com/736x/d5/82/5e/d5825ec732c450321c50d25bf6a0da43.jpg", "procedure": "1. Brush your hair to remove tangles.\n2. Apply a smoothing serum or gel to tame flyaways.\n3. Gather your hair at the crown of your head.\n4. Secure with an elastic band for a tight or loose ponytail.\n5. Wrap a strand of hair around the elastic to hide it.\n6. Use bobby pins to secure any loose strands.\n7. Tease the ponytail slightly for added volume.\n8. Apply hairspray to hold the style.\n9. Add accessories like scrunchies or ribbons for flair.\n10. Maintain by keeping hair healthy with regular trims."},
                {"name": "Pixie Cut", "img": "https://i.pinimg.com/736x/a1/bc/17/a1bc17ff4d4efc68b14914f5199b36ca.jpg", "procedure": "1. Begin with clean, damp hair.\n2. Apply a texturizing paste or wax.\n3. Use your fingers to create a messy, textured look.\n4. Finish with a light hold hairspray.\n5. Maintain with regular trims every 4-6 weeks.\n6. Keep the sides shorter than the top for a modern look.\n7. Style the top with different products for varied looks.\n8. Use a small amount of product to avoid weighing down fine hair.\n9. Add accessories like headbands or hair clips for variation.\n10. Protect your hair from heat damage with regular treatments."}
    ],
    "Heart": [
        {"name": "Chin-Length Bob", "img": "https://example.com/chin_bob.jpg", "procedure": "1. Consult a stylist for a chin-length bob cut.\n2. Wash and condition with hydrating products.\n3. Towel-dry gently and apply a heat protectant.\n4. Blow-dry your hair straight or with a slight curve.\n5. Use a flat iron for a sleek finish.\n6. Add texture with a texturizing spray.\n7. Style the ends inward or outward for variation.\n8. Finish with hairspray for hold.\n9. Maintain with trims every 4-6 weeks.\n10. Experiment with parting (center or side)."},
        {"name": "Side-Swept Bangs", "img": "https://i.pinimg.com/736x/ce/f6/38/cef63885a9d5caa7ed476feb8125b897.jpg", "procedure": "1. Consult a stylist to cut side-swept bangs.\n2. Wash and condition your hair as usual.\n3. Towel-dry gently and apply a heat protectant.\n4. Blow-dry the bangs to the side using a round brush.\n5. Use a flat iron to smooth or curl the bangs slightly.\n6. Style the rest of your hair as desired.\n7. Apply a light-hold spray to keep the bangs in place.\n8. Trim the bangs every 2-3 weeks to maintain length.\n9. Experiment with styling for different looks.\n10. Maintain healthy hair with regular trims."}
    ],
    "Diamond": [
        {"name": "Textured Waves", "img": "https://i.pinimg.com/736x/47/ff/26/47ff26597383773eec73f081de5786d7.jpg", "procedure": "1. Wash and condition with hydrating products.\n2. Towel-dry gently and apply a heat protectant.\n3. Blow-dry your hair with a diffuser for volume.\n4. Use a curling wand to create loose waves.\n5. Apply a texturizing spray for a beachy look.\n6. Scrunch the waves with your hands for added texture.\n7. Separate the waves with your fingers.\n8. Finish with hairspray to hold the waves.\n9. Maintain with trims every 6-8 weeks.\n10. Experiment with parting (center or side)."},
        {"name": "Deep Side Part", "img": "https://i.pinimg.com/736x/14/75/0e/14750e2354ef6f494034a7a7d9b4045d.jpg", "procedure": "1. Choose your side for the deep part.\n2. Wash and condition your hair as usual.\n3. Towel-dry gently and apply a heat protectant.\n4. Blow-dry your hair in the direction of the part.\n5. Use a flat iron to smooth or add waves.\n6. Apply a styling product like pomade or wax.\n7. Comb the hair smoothly to the side.\n8. Finish with hairspray for hold.\n9. Maintain with regular trims.\n10. Experiment with accessories like clips or headbands."}
    ],
    "Oblong": [
        {"name": "Long with Bangs", "img": "https://i.pinimg.com/736x/57/4f/52/574f526857b4ed6f0488d402cf94c2aa.jpg", "procedure": "1. Grow your hair to your desired length.\n2. Consult a stylist to add bangs (straight, side-swept, or wispy).\n3. Wash and condition with hydrating products.\n4. Towel-dry gently and apply a heat protectant.\n5. Blow-dry your hair straight or with waves.\n6. Style the bangs as desired (straight or side-swept).\n7. Use a flat iron or curling wand for added texture.\n8. Apply a light-hold spray to keep the bangs in place.\n9. Maintain with trims every 4-6 weeks.\n10. Experiment with parting (center or side)."},
        {"name": "Shoulder Length Waves", "img": "https://i.pinimg.com/736x/15/d6/5d/15d65d3879bd1e2a09fcbfbbc76f114c.jpg", "procedure": "1. Grow your hair to shoulder length.\n2. Consult a stylist to add soft layers for movement.\n3. Wash and condition with hydrating products.\n4. Towel-dry gently and apply a heat protectant.\n5. Blow-dry your hair with a round brush for volume.\n6. Use a curling wand to create loose waves.\n7. Apply a texturizing spray for a beachy look.\n8. Scrunch the waves with your hands for added texture.\n9. Finish with hairspray to hold the waves.\n10. Maintain with trims every 6-8 weeks."}
    ],
}

def distance(point1, point2):
    return np.hypot(point1.x - point2.x, point1.y - point2.y)

def calculate_face_shape(landmarks):
    landmark_indices = {
        "foreheadTop": 10,
        "chinBottom": 152,
        "leftCheek": 116,
        "rightCheek": 346,
        "leftJaw": 234,
        "rightJaw": 454,
        "leftTemple": 127,
        "rightTemple": 356,
    }

    get_point = lambda index: landmarks[index]

    points = {
        "forehead": get_point(landmark_indices["foreheadTop"]),
        "chin": get_point(landmark_indices["chinBottom"]),
        "leftCheek": get_point(landmark_indices["leftCheek"]),
        "rightCheek": get_point(landmark_indices["rightCheek"]),
        "leftJaw": get_point(landmark_indices["leftJaw"]),
        "rightJaw": get_point(landmark_indices["rightJaw"]),
        "leftTemple": get_point(landmark_indices["leftTemple"]),
        "rightTemple": get_point(landmark_indices["rightTemple"]),
    }

    face_height = distance(points["forehead"], points["chin"])
    cheekbone_width = distance(points["leftCheek"], points["rightCheek"])
    jaw_width = distance(points["leftJaw"], points["rightJaw"])
    forehead_width = distance(points["leftTemple"], points["rightTemple"])

    height_to_cheek_ratio = face_height / cheekbone_width
    jaw_to_cheek_ratio = jaw_width / cheekbone_width
    forehead_to_cheek_ratio = forehead_width / cheekbone_width

    if height_to_cheek_ratio > 1.45:
        return "Oblong" if jaw_to_cheek_ratio > 0.9 else "Diamond"
    elif jaw_to_cheek_ratio > 0.95:
        return "Square" if height_to_cheek_ratio < 1.1 else "Oval"
    elif forehead_to_cheek_ratio > 1.05:
        return "Heart" if jaw_to_cheek_ratio < 0.85 else "Round"
    elif jaw_to_cheek_ratio < 0.85 and forehead_to_cheek_ratio > 0.9:
        return "Diamond"
    elif height_to_cheek_ratio <= 1.1:
        return "Round"
    return "Oval"

def get_hair_region(image, landmarks):
    h, w, _ = image.shape

    forehead = landmarks[10]
    y = int(forehead.y * h)

    # Crop only middle 50% width
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    hair_region = image[int(y*0.3):y, x1:x2]

    return hair_region


def detect_hair_density(hair_region):
    gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)

    mean_intensity = np.mean(gray)

    if mean_intensity < 70:
        return "Dense"
    elif mean_intensity < 120:
        return "Medium"
    else:
        return "Low"


def detect_hair_type(hair_region):
    gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_ratio = np.sum(edges > 0) / (hair_region.shape[0] * hair_region.shape[1])

    if edge_ratio > 0.20:
        return "Curly"
    elif edge_ratio > 0.12:
        return "Wavy"
    else:
        return "Straight"

def get_recommendations(face_shape, gender):
    if gender == "men":
        return men_face_shapes.get(face_shape, [{"name": "No recommendations available.", "img": "", "procedure": ""}])
    else:
        return women_face_shapes.get(face_shape, [{"name": "No recommendations available.", "img": "", "procedure": ""}])

# Routes
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/men')
def men():
    return render_template('men.html')

@app.route('/women')
def women():
    return render_template('women.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        image_data = request.form['image']
        gender = request.form['gender']
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = face_mesh.process(image_cv2)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            face_shape = calculate_face_shape(landmarks)
            recommendations = get_recommendations(face_shape, gender)
            hair_region = get_hair_region(image_cv2, landmarks)
            hair_density = detect_hair_density(hair_region)
            hair_type = detect_hair_type(hair_region)

            return jsonify({
                'face_shape': face_shape,
                'hair_density': hair_density,
                'hair_type': hair_type,
                'recommendations': recommendations
            }) 
        else:
            return jsonify({'error': 'No face detected'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
