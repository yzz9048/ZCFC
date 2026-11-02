import os


class FileHandler:
    @staticmethod
    def set_folder(folder_name):
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        return folder_name
