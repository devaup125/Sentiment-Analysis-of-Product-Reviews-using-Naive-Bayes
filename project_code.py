import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
try:
    df = pd.read_csv("train.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("train.csv", encoding="latin1")
texts = df["selected_text"].astype(str)   
labels_raw = df["sentiment"].astype(str)  
le = LabelEncoder()
labels = le.fit_transform(labels_raw)    

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

def predict_sentiment():
    review = text_entry.get("1.0", tk.END).strip()
    if not review:
        messagebox.showwarning("Input Error", "Please enter a product review.")
        return
    progress['value'] = 55
    root.update()
    X_new = vectorizer.transform([review])
    pred_num = model.predict(X_new)[0]
    pred_class = le.inverse_transform([pred_num])[0]

    # Assign emoji and colors
    text_lower = review.lower()
    if pred_class == "positive":
        if any(word in text_lower for word in ['love', 'excellent', 'best', 'fantastic']):
            emoji = "😄"
            color = "#2e7d32"
            sentiment_text = "Very Positive"
        else:
            emoji = "😊"
            color = "#4caf50"
            sentiment_text = "Positive"
    elif pred_class == "negative":
        if any(word in text_lower for word in ['worst', 'terrible', 'awful', 'hate']):
            emoji = "😠"
            color = "#b71c1c"
            sentiment_text = "Very Negative"
        else:
            emoji = "😕"
            color = "#e53935"
            sentiment_text = "Negative"
    else: # neutral
        emoji = "😐"
        color = "#ff9800"
        sentiment_text = "Neutral"

    result_label.config(text=f"Sentiment: {emoji} {sentiment_text}", fg=color)
    progress['value'] = 100
    root.after(800, lambda: progress.config(value=0))

def reset_fields():
    text_entry.delete("1.0", tk.END)
    result_label.config(text="Sentiment: ", fg="#000000")
    progress['value'] = 0

def on_enter(e):
    analyze_btn['bg'] = "#00594d"
def on_leave(e):
    analyze_btn['bg'] = "#00897b"
def reset_enter(e):
    reset_btn['bg'] = "#ab000d"
def reset_leave(e):
    reset_btn['bg'] = "#e53935"

root = tk.Tk()
root.title("Product Review Sentiment Analysis")
root.geometry("500x350")
root.resizable(False, False)
root.configure(bg="#f9f9f9")

frame = tk.Frame(root, bg="#d4edda", bd=2)
frame.place(x=30, y=60, width=440, height=230)

title_label = tk.Label(root, text="Product Review Sentiment", font=("Verdana", 18, "bold"), bg="#f9f9f9", fg="#1a237e")
title_label.place(x=80, y=10)

prompt = tk.Label(frame, text="Enter product review:", font=("Arial", 13, 'bold'), bg="#004d40", fg="white")
prompt.place(x=10, y=10)

text_entry = tk.Text(frame, height=3, width=42, font=("Arial", 12), bg='white', highlightbackground="#c8e6c9", highlightthickness=1)
text_entry.place(x=15, y=40)

scroll = ttk.Scrollbar(frame, orient='vertical', command=text_entry.yview)
text_entry.config(yscrollcommand=scroll.set)
scroll.place(x=388, y=40, height=68)

analyze_btn = tk.Button(frame, text="Analyze Sentiment", command=predict_sentiment, font=("Arial", 12), bg="#00897b", fg="white", relief=tk.RAISED, cursor="hand2", width=15)
analyze_btn.place(x=60, y=120)
analyze_btn.bind("<Enter>", on_enter)
analyze_btn.bind("<Leave>", on_leave)

reset_btn = tk.Button(frame, text="Reset", command=reset_fields, font=("Arial", 12), bg="#e53935", fg="white", relief=tk.RAISED, cursor="hand2", width=10)
reset_btn.place(x=230, y=120)
reset_btn.bind("<Enter>", reset_enter)
reset_btn.bind("<Leave>", reset_leave)

style = ttk.Style()
style.theme_use('default')
style.configure("green.Horizontal.TProgressbar", foreground="#00897b", background="#4caf50")
progress = ttk.Progressbar(frame, length=300, mode='determinate', style="green.Horizontal.TProgressbar")
progress.place(x=40, y=170)

result_label = tk.Label(frame, text="Sentiment: ", font=("Arial", 15), bg="#d4edda", fg="#000000")
result_label.place(x=15, y=200)

root.mainloop()
