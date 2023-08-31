import os
import shutil


def copy_and_rename_files(source_folder, destination_folder):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件夹中的文件列表
    source_files = os.listdir(source_folder)

    # 遍历并复制文件
    for index, file_name in enumerate(source_files):
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_name = str(index+5076) + os.path.splitext(file_name)[1]
        destination_file_path = os.path.join(destination_folder, destination_file_name)
        shutil.copy2(source_file_path, destination_file_path)


if __name__ == "__main__":
    source_folder = r"F:\archive\Sugar beet"
    destination_folder = r"F:\archive\data"
    copy_and_rename_files(source_folder, destination_folder)
