import json
import os


def load_data_to_one_file(json_folder):
    """
        Загружает данные из JSON-файлов в указанной папке, извлекает координаты суставов руки
        и соответствующие метки классов, затем сохраняет их в файл dataset.json.

        Аргументы:
        json_folder (str): Путь к папке, содержащей JSON-файлы с данными жестов.

        Выход:
        Создает и сохраняет файл dataset.json с координатами (X) и метками классов (Y).
        """
    if os.path.exists("dataset.json"):  # Проверяем, существует ли dataset.json
        print("Файл dataset.json уже существует. Пропускаем загрузку.")  # Сообщаем, что загрузка не требуется
    else:
        x, y = [], []  # Списки для хранения входных данных (координаты) и выходных меток (классы)

        # Создаем словарь, который сопоставляет буквам числа (метки классов)
        label_map = {"Ничего": 0}  # "Ничего" обозначает жест, не соответствующий буквам, и имеет метку 0
        label_map.update({letter: i + 1 for i, letter in enumerate("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")})
        print(label_map)  # Выводим на экран соответствие букв и их индексов

        # Проходим по всем 34 категориям (33 буквы + 1 "Ничего")
        for i in range(34):
            path = os.path.join(json_folder, str(i))  # Формируем путь к папке с файлами текущего класса

            if not os.path.exists(path):  # Проверяем, существует ли папка
                print(f"Папка {path} не найдена, пропускаем...")
                continue  # Если папка отсутствует, переходим к следующему классу

            for filename in os.listdir(path):  # Перебираем все файлы в текущей папке
                if not filename.endswith(".json"):  # Проверяем, является ли файл JSON
                    continue  # Если это не JSON, пропускаем файл

                # Полный путь к JSON-файлу
                file_path = os.path.join(path, filename)

                # Открываем JSON-файл и загружаем его содержимое
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)  # Загружаем JSON-данные

                    # Перебираем все буквы в JSON-файле (по идее, там должна быть только одна)
                    for letter, points in data.items():
                        label = label_map.get(letter, 0)  # Получаем метку класса из словаря, если буквы нет — 0
                        # Извлекаем координаты 21 сустава руки и создаем список признаков
                        features = [points[str(j)][key] for j in range(21) for key in ["x", "y", "z"]]

                        x.append(features)  # Добавляем координаты в общий список X
                        y.append(label)  # Добавляем метку класса в общий список Y

                        print(f'Файл {file_path} загружен')  # Выводим сообщение о загруженном файле

    # Создаем словарь с извлеченными данными
    data = {"X": x, "Y": y}

    # Записываем данные в файл dataset.json
    with open("dataset.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)  # Сохраняем в формате JSON с отступами для читаемости

    print("Данные успешно сохранены в dataset.json")  # Выводим сообщение об успешном сохранении


# Основной блок кода, выполняемый при запуске скрипта
if __name__ == "__main__":
    load_data_to_one_file(os.path.abspath(os.path.join(os.getcwd(), "..", "references")))
