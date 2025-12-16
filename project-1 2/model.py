import numpy as np
import re

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() # словарь, содержащий все уникальные слова из набора train
        self.spam = {} # словарь, содержащий частоту слов в спам-сообщениях из набора данных train.
        self.ham = {} # словарь, содержащий частоту слов в не спам-сообщениях из набора данных train.
        self.alpha = alpha # сглаживание
        self.label2num = None # словарь, используемый для преобразования меток в числа
        self.num2label = None # словарь, используемый для преобразования числа в метки
        self.Nvoc = None # общее количество уникальных слов в наборе данных train
        self.Nspam = None # общее количество уникальных слов в спам-сообщениях в наборе данных train
        self.Nham = None # общее количество уникальных слов в не спам-сообщениях в наборе данных train
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def fit(self, dataset):
        '''
        dataset - объект класса Dataset
        Функция использует входной аргумент "dataset", 
        чтобы заполнить все атрибуты данного класса.
        '''
        # Начало вашего кода
        # Сохранить данные
        self._train_X, self._train_y = dataset.train
        self._val_X, self._val_y = dataset.val
        self._test_X, self._test_y = dataset.test
        self.label2num = dataset.label2num
        self.num2label = dataset.num2label
        
        # Подсчитать частоты слов для спама и не-спама
        spam_count = 0
        ham_count = 0
        
        for i, message in enumerate(self._train_X):
            words = message.split()
            label = self._train_y[i]
            
            if label == "spam":
                spam_count += 1
                for word in words:
                    self.vocab.add(word)
                    if word not in self.spam:
                        self.spam[word] = 0
                    self.spam[word] += 1
            else:
                ham_count += 1
                for word in words:
                    self.vocab.add(word)
                    if word not in self.ham:
                        self.ham[word] = 0
                    self.ham[word] += 1
        
        # Вычислить общее количество слов
        self.Nvoc = len(self.vocab)
        self.Nspam = sum(self.spam.values())
        self.Nham = sum(self.ham.values())
        
        # Сохранить вероятности классов
        total = spam_count + ham_count
        self.p_spam = spam_count / total
        self.p_ham = ham_count / total
        # Конец вашего кода
    
    def inference(self, message):
        '''
        Функция принимает одно сообщение и, используя наивный байесовский алгоритм, определяет его как спам / не спам.
        '''
        # Начало вашего кода
        # Очистить сообщение
        text = str(message).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = ' '.join(text.split())
        words = text.split()
        
        # Вычислить вероятности
        log_pspam = np.log(self.p_spam)
        log_pham = np.log(self.p_ham)
        
        for word in words:
            # Вероятность слова в спаме
            count_spam = self.spam.get(word, 0)
            p_word_spam = (count_spam + self.alpha) / (self.Nspam + self.alpha * self.Nvoc)
            log_pspam += np.log(p_word_spam)
            
            # Вероятность слова в не-спаме
            count_ham = self.ham.get(word, 0)
            p_word_ham = (count_ham + self.alpha) / (self.Nham + self.alpha * self.Nvoc)
            log_pham += np.log(p_word_ham)
        
        pspam = np.exp(log_pspam)
        pham = np.exp(log_pham)
        # Конец вашего кода
        if pspam > pham:
            return "spam"
        return "ham"
    
    def validation(self):
        '''
        Функция предсказывает метки сообщений из набора данных validation,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        correct = 0
        total = len(self._val_y)
        
        for i in range(total):
            pred = self.inference(self._val_X[i])
            if pred == self._val_y[i]:
                correct += 1
        
        val_acc = correct / total
        # Конец вашего кода
        return val_acc 

    def test(self):
        '''
        Функция предсказывает метки сообщений из набора данных test,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        correct = 0
        total = len(self._test_y)
        
        for i in range(total):
            pred = self.inference(self._test_X[i])
            if pred == self._test_y[i]:
                correct += 1
        
        test_acc = correct / total
        # Конец вашего кода
        return test_acc


