import pandas as pd
import numpy as np
from dataset import Dataset
from model import Model

# Загрузить данные
print("Загрузка данных...")
data = pd.read_csv('spam.csv', encoding='latin-1')
X = data['v2'].values
y = data['v1'].values

# Создать датасет
print("Подготовка данных...")
dataset = Dataset(X, y)
dataset.split_dataset(val=0.1, test=0.1)

# Обучить модель
print("Обучение модели...")
model = Model(alpha=1)
model.fit(dataset)

# Проверить точность
val_acc = model.validation()
test_acc = model.test()
print(f"\nТочность на валидации: {val_acc:.4f}")
print(f"Точность на тесте: {test_acc:.4f}\n")

# Тестирование через терминал
print("=" * 50)
print("Тест классификатора спама")
print("Введите текст сообщения (или 'exit' для выхода)")
print("=" * 50)

while True:
    text = input("\nВведите текст: ")
    if text.lower() == 'exit':
        break
    
    if text.strip() == '':
        print("Пожалуйста, введите текст!")
        continue
    
    result = model.inference(text)
    if result == "spam":
        print("Результат: СПАМ ⚠️")
    else:
        print("Результат: НЕ СПАМ ✓")

print("\nДо свидания!")

