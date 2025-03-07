Используемый датасет -> [ddos](https://www.kaggle.com/datasets/devendra416/ddos-datasets)

---

### **1. Файл FOREST.py**
**Цель:** Анализ важности признаков в данных сетевого трафика для классификации DDoS-атак с использованием модели Random Forest.

#### **Этапы выполнения:**
1. **Импорт библиотек:**
   - `numpy`, `pandas` для обработки данных.
   - `sklearn.ensemble.RandomForestClassifier` для построения модели.
   - `time` для замера времени выполнения.

2. **Загрузка данных:**
   - Чтение файла `final_dataset.csv` с оптимизированными типами данных (для экономии памяти).
   - Используется только 100 строк (`nrows=100`) для быстрого тестирования.
   - Удаление избыточных колонок:
     - `Src IP`, `Dst IP` (категориальные данные с высокой кардинальностью).
     - Колонки с одним значением (напр., `Fwd Byts/b Avg`).

3. **Предобработка:**
   - Замена бесконечных значений (`inf`, `-inf`) на `NaN` и их заполнение нулями.
   - Преобразование метки `Label` в бинарный формат: `Benign` → `1`, `ddos` → `0`.

4. **Обучение модели:**
   - Создание модели Random Forest с 250 деревьями.
   - Обучение на данных и расчет важности признаков (`feature_importances_`).
   - Вывод топ-20 важных признаков (напр., `Fwd Pkt Len Max`, `Dst Port`).

5. **Результаты:**
   - Сохранение списка важных признаков в `importance_list_all_data.csv`.
   - Вывод времени выполнения скрипта.

---

### **2. Файл neural-net.py**
**Цель:** Классификация DDoS-атак с использованием нейронной сети на базе LSTM и интеграция с платформой Neptune для мониторинга.

#### **Этапы выполнения:**
1. **Импорт библиотек:**
   - `tensorflow`, `keras` для построения нейросети.
   - `neptune` для логирования экспериментов.
   - Обработка данных через `pandas` и `numpy`.

2. **Настройка Neptune:**
   - Инициализация сессии с параметрами модели (learning rate, epochs, batch size).
   - Логирование метрик в реальном времени.

3. **Загрузка данных:**
   - Чтение полного датасета `final_dataset.csv` (без ограничения строк).
   - Удаление колонок `Src IP`, `Dst IP`.
   - Аналогичная предобработка (замена `inf`, `NaN`).

4. **Подготовка данных:**
   - Выбор признаков из списка `logfeatures` (напр., `Dst Port`, `Flow IAT Mean`).
   - Разделение данных на тренировочные (`80%`) и тестовые (`10%` + валидационные `10%`).

5. **Построение модели:**
   - Архитектура:
     - Три слоя LSTM (10 нейронов каждый).
     - Выходной слой с сигмоидной активацией для бинарной классификации.
   - Компиляция с оптимизатором Adam и функцией потерь `binary_crossentropy`.

6. **Обучение и оценка:**
   - Обучение с логированием в Neptune.
   - Расчет точности на тестовых данных (метрика `accuracy`).
   - Построение матрицы ошибок для анализа ложных срабатываний (`dc_mf_fn`).

---

### **Рекомендации:**
1. Для улучшения производительности в `FOREST.py` можно увеличить `nrows` для анализа на полных данных.
2. В `neural-net.py` стоит проверить совместимость архитектуры LSTM с данными (возможно, требуется ресемплирование в 3D-тензор).
3. Добавить кросс-валидацию и оптимизацию гиперпараметров (напр., через `GridSearchCV` или `KerasTuner`).

**Время выполнения FOREST.py:** Указано в конце скрипта (зависит от объема данных, на 12 млн объектов было около 2-х дней).  
**Точность neural-net.py:** Выводится через `test_acc` (было., `>95%`).
