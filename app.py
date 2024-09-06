from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the TF-IDF Vectorizer and the trained model
loaded_vect = joblib.load('tfidf_vectorizer.pkl')
loaded_model = load_model('toxic_comment_model.h5')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict_toxicity():
    # Get JSON data from the request
    data = request.json
    comments = data.get('comments', [])

    # Check if comments is a string, convert to list
    if isinstance(comments, str):
        comments = [comments]

    # Check if comments list is empty
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    # Preprocess the comments using the loaded TF-IDF Vectorizer
    comments_tfidf = loaded_vect.transform(comments)

    # Predict using the loaded model
    predictions = (loaded_model.predict(comments_tfidf.toarray()) > 0.5).astype(int)

    # Prepare results
    results = [
        {"comment": comment, "toxic": bool(prediction)}
        for comment, prediction in zip(comments, predictions)
    ]

    # Return predictions as JSON
    return jsonify(results)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
