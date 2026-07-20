# house_prices

Проект для педсказания стоимости дома на основе табличных данных


## Ключевые особенности

Classic ML: Использование библиотеки `scikit-learn` для построения базовых моделей (Random Forest, Gradient Boosting и др.).
Deep Learning: Полносвязная нейронная сеть, реализованная на фреймворке `PyTorch`.


## Стек технологий

Язык программирования:Python 3.14.3
Классический ML: `scikit-learn`, `pandas`, `numpy`
Глубокое обучение: `torch` (PyTorch)


## Инструкция по установке и запуску

### 1. Клонирование репозитория и переход в папку проекта

git clone https://github.com/gdjdjdhhdj9/house_prices.git
cd house_prices


### 2. Создание и активация виртуального окружения (venv)

python -m venv .venv
.venv\Scripts\activate


### 3. Установка зависимостей

python.exe -m pip install --upgrade pip
pip install -r requirements.txt


### 4. Выбор и запуск модели

python3 main_ml.py  # если хотим использовать класический ML
python3 main_nn.py  # если хотим использовать глубокое обучение


## Структура проекта

1. data - Папка с наборами данных (train.csv, test.csv) и выводами.

2. models - Папка с лучшими моделями (ml, nn).

3. notebooks - Папка с Jupyter-ноутбуками

4. config.py - Конфигурационный файл (пути к файлам, гиперпараметры)

5. feature_engineering.py - Создание новых признаков , генерация фичей

6. preprocessing.py - препроцессор(где происходит кодирование данных и тд)

7. models_list.py - Список конфигураций классических моделей ML

8. train_ml.py - Скрипт для обучения классического ML (scikit-learn)

9. main_ml.py - Основной файл для запуска обучения и инференса классического ML

10. nn_data_preparation.py - Подготовка датасетов и DataLoader для PyTorch

11. nn_model.py - Архитектура полносвязной нейронной сети

12. train_nn.py - Скрипт для обучения нейросети

13. main_nn.py - Основной файл для запуска обучения и инференса нейросети

14. metrics.py - Функции расчета метрик качества

15. requirements.txt - Список внешних зависимостей проекта

16. README.md - Документация проекта
