from flask import Flask, request, jsonify
from kafka import KafkaConsumer, KafkaProducer
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from threading import Thread

# Kafka config
KAFKA_BROKER = '192.168.120.131:9092'
SANITIZE_TOPIC = 'sanitize-comments'
RESULT_TOPIC = 'sanitized-comments'

app = Flask(__name__)

# Load model and tokenizer
model_name = "Model_Training"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Kafka producer
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def predict_user_input(input_text, model, tokenizer, device):
    user_input = [input_text]
    user_encodings = tokenizer(
        user_input, truncation=True, padding=True, return_tensors="pt"
    )
    input_ids = user_encodings['input_ids'].to(device)
    attention_mask = user_encodings['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.sigmoid(logits)

    return (predictions.cpu().numpy() > 0.5).astype(int)[0]

def sanitize_comment(text):
    def mask_word(word):
        return '*' * len(word)

    words = text.split()
    sanitized_words = [
        mask_word(word) if any(predict_user_input(word, model, tokenizer, device)) else word
        for word in words
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

        # Update sanitizedText field
        comment_data['sanitizedText'] = sanitized_text

        # Send response to sanitized-comments topic
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
        predicted_labels = predict_user_input(comment_text, model, tokenizer, device)
        is_toxic = any(predicted_labels)
        result = {"comment": comment_text, "toxic": is_toxic}
        results.append(result)

    return jsonify(results)

if __name__ == '__main__':
    consumer_thread = Thread(target=consume_comments)
    consumer_thread.start()

    app.run(host='0.0.0.0', port=5609)
