from sentence_transformers import InputExample
import csv

train_examples = []
with open("./data/demo_golden_1200.txt", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if len(row) != 2:
            continue
        train_examples.append(InputExample(texts=[row[0], row[1]], label=1.0))

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('LaBSE')

from torch.utils.data import DataLoader
from sentence_transformers import losses

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,  # or more
    warmup_steps=100,
    output_path='./fine_tuned_model'
)
