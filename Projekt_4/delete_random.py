import os
import random


def delete_random_files(directory, desired_count):
    # Lista wszsytkich plików w katalogu
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Kalkulacja ile plików trzeba usunąć
    num_files_to_delete = len(files) - desired_count

    if num_files_to_delete <= 0:
        print("Katalog ma juz tyle plikow ile potrzeba lub mniej. Nie ma potrzeby usuwania plikow.")
        return

    # Losowe wybranie plików do usunięcia
    files_to_delete = random.sample(files, num_files_to_delete)

    # Usuniecie wybranych plików
    for file in files_to_delete:
        file_path = os.path.join(directory, file)
        os.remove(file_path)
        print(f"Usunieto: {file_path}")

    print(f"Usunieto {num_files_to_delete} plikow. Obecna liczba plikow: {desired_count}")


# Uzycie
directory_path = 'D:/PJATK/Semestr_6/WMA---Wizja-maszynowa/Projekt_4/output_frames_me3'
desired_file_count = 5921

delete_random_files(directory_path, desired_file_count)
