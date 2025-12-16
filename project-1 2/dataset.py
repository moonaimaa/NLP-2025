import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X # сообщения 
        self._y = y # метки ["spam", "ham"]
        self.train = None # кортеж из (X_train, y_train)
        self.val = None # кортеж из (X_val, y_val)
        self.test = None # кортеж из (X_test, y_test)
        self.label2num = {} # словарь, используемый для преобразования меток в числа
        self.num2label = {} # словарь, используемый для преобразования числа в метки
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        '''
        Функция очистки сообщения и преобразования меток в числа.
        '''
        # Начало вашего кода
        # Очистка текста
        cleaned_x = []
        for text in self._x:
            if text is None:
                text = ""
            # Привести к нижнему регистру
            text = str(text).lower()
            # Оставить только буквы и пробелы
            text = re.sub(r'[^a-z\s]', ' ', text)
            # Удалить лишние пробелы
            text = ' '.join(text.split())
            cleaned_x.append(text)
        self._x = cleaned_x
        
        # Преобразование меток в числа
        unique_labels = list(set(self._y))
        self.label2num = {label: i for i, label in enumerate(unique_labels)}
        self.num2label = {i: label for i, label in enumerate(unique_labels)}
        # Конец вашего кода

    def split_dataset(self, val=0.1, test=0.1):
        '''
        Функция, которая разбивает набор данных на наборы train-validation-test.
        '''
        # Начало вашего кода
        n = len(self._x)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # Вычислить размеры
        test_size = int(n * test)
        val_size = int(n * val)
        
        # Разделить индексы
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        train_indices = indices[test_size + val_size:]
        
        # Разделить данные
        self.test = (np.array(self._x)[test_indices].tolist(), 
                     np.array(self._y)[test_indices].tolist())
        self.val = (np.array(self._x)[val_indices].tolist(), 
                    np.array(self._y)[val_indices].tolist())
        self.train = (np.array(self._x)[train_indices].tolist(), 
                      np.array(self._y)[train_indices].tolist())
        # Конец вашего кода
