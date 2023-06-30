import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

values = ["negatif", "positive"]

def predict(text):
    probabilities = transformers_model(text)[0]
    d = {}
    d[probabilities['label']] = probabilities['score']
    if probabilities['label']=="POSITIVE" :
        d["NEGATIVE"] =  1 - probabilities["score"]
    else:
        d["POSITIVE"] = 1 - probabilities["score"]
    return d

if __name__=='__main__':
    tokenizer_path = 'pretrained-tokenizer'
    model_path = 'pretrained-model'
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    transformers_model = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    gr.Interface(fn=predict, 
                inputs="text", 
                outputs=gr.components.Label(num_top_classes=2),
                live=True,
                description="Write a comment and it will be evaluated.",
                ).launch(debug=True, share=True);
