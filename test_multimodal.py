from mymodel import MyAIModel
from ai.vision import PIL_AVAILABLE

print("PIL available for real photos:", PIL_AVAILABLE)
print()

model = MyAIModel(name="RohanAI")

model.teach_classification("shape_question", ["what shape is this", "describe the image", "what do you see"])
model.teach_classification("greeting", ["hi", "hello"])
model.train_classifier(epochs=300)

x1 = [[1,0,0,0,0,0,0,1],[0,1,0,0,0,0,1,0],[0,0,1,0,0,1,0,0],[0,0,0,1,1,0,0,0],
      [0,0,0,1,1,0,0,0],[0,0,1,0,0,1,0,0],[0,1,0,0,0,0,1,0],[1,0,0,0,0,0,0,1]]
x2 = [[1,0,0,0,0,0,1,0],[0,1,0,0,0,1,0,0],[0,0,1,0,1,0,0,0],[0,0,0,1,1,0,0,0],
      [0,0,0,1,1,0,0,0],[0,0,1,0,0,1,0,0],[0,1,0,0,0,0,1,0],[1,0,0,0,0,0,0,1]]
o1 = [[0,0,1,1,1,1,0,0],[0,1,0,0,0,0,1,0],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],
      [1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1],[0,1,0,0,0,0,1,0],[0,0,1,1,1,1,0,0]]
o2 = [[0,0,1,1,1,1,0,0],[0,1,0,0,0,0,1,0],[1,0,0,1,1,0,0,1],[1,0,0,0,0,0,0,1],
      [1,0,0,0,0,0,0,1],[1,0,0,1,1,0,0,1],[0,1,0,0,0,0,1,0],[0,0,1,1,1,1,0,0]]

model.teach_vision("X", x1)
model.teach_vision("X", x2)
model.teach_vision("O", o1)
model.teach_vision("O", o2)
model.train_vision(epochs=300)

print(model)
print()
print("=== Multimodal understanding (text + image together) ===")
result = model.understand_multimodal(text="what shape is this", image=x1)
print(result)
print()

model.save("rohanai_multimodal.json")
print("Model saved to rohanai_multimodal.json")

print()
print("=== Loading fresh and testing again ===")
loaded = MyAIModel.load("rohanai_multimodal.json")
print(loaded)
print(loaded.understand_multimodal(text="what shape is this", image=x1))
