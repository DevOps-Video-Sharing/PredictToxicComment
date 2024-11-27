from flask import Flask, request, jsonify
from kafka import KafkaConsumer, KafkaProducer
from tensorflow.keras.models import load_model
import joblib
import json
import re

# Kafka config
KAFKA_BROKER = 'localhost:9092'
SANITIZE_TOPIC = 'sanitize-comments'
RESULT_TOPIC = 'sanitized-comments'

app = Flask(__name__)

loaded_vect = joblib.load('tfidf_vectorizer.pkl')
loaded_model = load_model('toxic_comment_model.h5')

# Kafka producer
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def predict_toxicity(comment_text):
    comment_tfidf = loaded_vect.transform([comment_text])
    prediction = (loaded_model.predict(comment_tfidf.toarray()) > 0.5).astype(int)
    return bool(prediction[0][0])


def sanitize_comment(text):
    def mask_word(word):
        # Trả về số lượng dấu '*' bằng độ dài của từ
        return '*' * len(word)

    words = text.split()
    sanitized_words = [
        mask_word(word) if predict_toxicity(word) else word for word in words
    ]
    return " ".join(sanitized_words)


def consume_comments():
    consumer = KafkaConsumer(
        SANITIZE_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='flask-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    for message in consumer:
        comment_data = message.value
        comment_text = comment_data.get('text', '')
        sanitized_text = sanitize_comment(comment_text)

        # Cập nhật trường sanitizedText
        comment_data['sanitizedText'] = sanitized_text

        # Gửi phản hồi qua Kafka topic sanitized-comments
        producer.send(RESULT_TOPIC, comment_data)
        

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments', [])

    if isinstance(comments, str):
        comments = [comments]

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    results = []
    for comment_text in comments:
        is_toxic = predict_toxicity(comment_text)
        result = {"comment": comment_text, "toxic": is_toxic}
        results.append(result)

    return jsonify(results)

if __name__ == '__main__':
    from threading import Thread
    consumer_thread = Thread(target=consume_comments)
    consumer_thread.start()

    app.run(host='0.0.0.0', port=5000)
