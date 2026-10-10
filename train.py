"""Train the intent classification neural network (a few seconds on CPU).

Pipeline: sentence -> MiniLM transformer encoder (pretrained, frozen) -> 384-dim
embedding -> IntentNet (3-layer feed-forward network trained here) -> intent.
"""
import json
import random

import numpy as np
import torch
import torch.nn as nn

from intent_model import IntentNet
from nlp_utils import EMBED_DIM, embed

HIDDEN = 256
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

with open("intents.json", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

tags = [i["tag"] for i in intents]
sentences, labels = [], []
for intent in intents:
    for p in intent["patterns"]:
        sentences.append(p)
        labels.append(tags.index(intent["tag"]))

# Unseen paraphrases (never used for training) to measure real generalisation.
TEST_SET = [
    ("hiya, anybody here?", "greeting"), ("good day to you", "greeting"),
    ("ok i'm off now", "goodbye"), ("later, take care", "goodbye"),
    ("many thanks for that", "thanks"), ("that was really useful", "thanks"),
    ("how's everything with you", "how_are_you"),
    ("doing great today", "user_feeling_good"),
    ("i feel really down", "user_feeling_bad"), ("today was horrible", "user_feeling_bad"),
    ("may i know your name", "name"), ("who am i talking to", "name"),
    ("who programmed you", "creator"),
    ("what are you capable of", "capabilities"), ("what kind of things can you help with", "capabilities"),
    ("can you tell me what time it is", "time"), ("what's the clock say", "time"),
    ("what's the date today", "date"), ("which date is it today", "date"),
    ("crack a joke", "joke"), ("tell me something funny", "joke"),
    ("share a cool fact", "fun_fact"),
    ("how do you recognize what i say", "speech_recognition"),
    ("what kind of ai are you", "deep_learning"), ("how does your brain work", "deep_learning"),
    ("will it snow tomorrow", "weather"), ("what's the temperature outside", "weather"),
    ("what's your age", "age"),
    ("what do you enjoy doing", "hobby"),
    ("you're brilliant", "compliment"), ("great work bot", "compliment"),
    ("you are terrible", "insult"),
    ("how should i revise for my test", "study_tips"), ("i can't focus on studying", "study_tips"),
    ("i feel like quitting", "motivation"), ("give me some inspiration", "motivation"),
]

print("Encoding sentences with MiniLM...")
X = torch.tensor(embed(sentences))
y = torch.tensor(labels)
X_test = torch.tensor(embed([s for s, _ in TEST_SET]))
y_test = torch.tensor([tags.index(t) for _, t in TEST_SET])

model = IntentNet(EMBED_DIM, HIDDEN, len(tags))
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()


def accuracy(Xs, ys):
    model.eval()
    with torch.no_grad():
        return (model(Xs).argmax(1) == ys).float().mean().item()


for epoch in range(1, 61):
    model.train()
    perm = torch.randperm(len(X))
    for i in range(0, len(X), 16):
        b = perm[i:i + 16]
        # small gaussian noise on embeddings = cheap data augmentation
        xb = X[b] + 0.02 * torch.randn_like(X[b])
        optimizer.zero_grad()
        loss = loss_fn(model(xb), y[b])
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}  "
              f"train acc {accuracy(X, y):.3f}  unseen-test acc {accuracy(X_test, y_test):.3f}")

model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(1)
for (s, t), p in zip(TEST_SET, preds):
    if tags[p] != t:
        print(f"  miss: '{s}' -> {tags[p]} (expected {t})")
print(f"\nFinal: train acc {accuracy(X, y):.3f} | unseen-test acc {accuracy(X_test, y_test):.3f}")

torch.save({"state_dict": model.state_dict(), "tags": tags, "hidden_size": HIDDEN}, "model.pth")
# Export weights for the PyTorch-free server (chatbot.py runs the same network in NumPy)
sd = {k: v.numpy() for k, v in model.state_dict().items()}
np.savez("model.npz", W1=sd["net.0.weight"], b1=sd["net.0.bias"], W2=sd["net.3.weight"],
         b2=sd["net.3.bias"], W3=sd["net.6.weight"], b3=sd["net.6.bias"], tags=np.array(tags))
print(f"saved model.pth + model.npz  ({len(tags)} intents, {len(sentences)} training samples)")
