import logging
import time
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.ERROR)

def process_text(text, sentiment_pipeline):
    # Ensure 'text' is a valid string
    if isinstance(text, str):
        try:
            # Get sentiment for the entire text
            sentiment_result = sentiment_pipeline(text)[0]

            # Extract sentiment label and polarity
            overall_sentiment_label = sentiment_result['label']
            if overall_sentiment_label == 'positive':
                overall_polarity = sentiment_result['score'] - 0.05
                overall_sentiment_label = 'Positive'
            elif overall_sentiment_label == 'negative':
                overall_polarity = sentiment_result['score'] * (-1) + 0.05
                overall_sentiment_label = 'Negative'
            else:
                overall_polarity = sentiment_result['score'] - 0.05
                overall_sentiment_label = 'Neutral'

            return overall_sentiment_label, round(overall_polarity, 2)
        
        except Exception as e:
            logging.error(f"Error processing text: {e}")
            print(f"Error processing text: {e}")
            print(f"Problematic text: {text}")
            return 'No Valid Predictions', 0.0

    else:
        return 'Invalid Text', 0.0

def analyze_sentiment(input_text):
    start_time = time.time()

    # Create sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="soleimanian/financial-roberta-large-sentiment", max_length=512, truncation=True)

    # Process the single input text
    sentiment, polarity = process_text(input_text, sentiment_pipeline)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")

    return sentiment, polarity

# Example usage
if __name__ == "__main__":
    # Multi-line input text
    input_text = '''According to the complaint filed by Saini’s wife, the accused arrived at their residence around 11:48 pm on October 30 and began arguing with Saini. The confrontation escalated as the accused forced their way inside and attacked Saini with a sharp-edged weapon and a country-made pistol. When Shahid Khan attempted to protect him, he was also assaulted with a sharp weapon. The attackers then beat up one Harshit Kumar, who was also present in the room.

As villagers reached the spot upon hearing the commotion, the assailants managed to escape.

The injured persons were taken to the hospital, where doctors declared Saini dead. The other two were admitted for treatment.
Khan told mediapersons that he and Saini were having dinner when Saini received a phone call. A few minutes later, the accused arrived, and a heated argument broke out between them at the gate of Saini’s residence. The confrontation escalated as the accused forcibly entered the house and attacked Saini and the others.

The police have registered an FIR against nine named and six unidentified persons at the Kotwali police station.'''

    # Analyze sentiment and polarity
    sentiment, polarity = analyze_sentiment(input_text)

    # Display results
    print(f"\nSentiment: {sentiment}")
    print(f"Polarity: {polarity}")
