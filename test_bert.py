########################### AI testing ##########################



from services.bert_service import predict_news

print("=" * 40)
print("Testing DistilBERT")
print("=" * 40)

text = "NASA announces discovery of water on Mars."

prediction, confidence = predict_news(text)

print(f"\nNews: {text}")
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}%")