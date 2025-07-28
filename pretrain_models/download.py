import gdown
import re

# 读取文件ID列表
with open('file_ids.txt', 'r') as file:
    for line in file:
        url = line.strip()

        # 跳过以 # 开头的行
        if url.startswith('#'):
            continue

        # 使用正则表达式提取文件ID
        file_id = re.search(r'https://drive\.google\.com/file/d/([^/]+)/', url)
        if file_id:
            print(f"从 URL 提取文件 ID: {file_id.group(1)}")
            file_id = file_id.group(1)
            download_url = f"https://drive.google.com/uc?id={file_id}"
            # gdown.download(download_url, quiet=False,use_cookies=False)
            try:
                # 尝试下载文件
                gdown.download(download_url, quiet=False)
            except PermissionError as e:
                print(f"下载文件时发生权限错误: {e}. 将继续下载下一个文件。")
            except Exception as e:
                print(f"下载文件时发生错误: {e}. 将继续下载下一个文件。")
        else:
            print(f"无法从 URL 提取文件 ID: {url}")

# gdown https://drive.google.com/uc?id=标识符