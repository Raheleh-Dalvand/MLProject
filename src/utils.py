import os
import sys
import pickle

from src.exception import CustomException


def save_object(file_path: str, obj) -> None:
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)
        with open(file_path, "wb") as file_object:
            pickle.dump(obj, file_object)
        print(f"✅ Pickle file saved at: {file_path}")
    except Exception as error:
        raise CustomException(error, sys)