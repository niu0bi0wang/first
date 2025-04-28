import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# --- 配置区 ---
# 1. 定义脚本需要的权限范围。对于 Drive 文件访问，通常用这个。
SCOPES = ['https://www.googleapis.com/auth/drive'] # 只读权限足够下载
# 如果脚本需要修改或删除文件，用 ''https://www.googleapis.com/auth/drive.readonly

# 2. 指定你的 OAuth 2.0 凭据文件路径 (刚才下载的 JSON 文件)
CLIENT_SECRET_FILE = r"C:\Users\27268\Desktop\item\client_secret_949540819720-ne0512dpcnu3gj7pcri4kt1c98b1re60.apps.googleusercontent.com.json" # !!! 替换成你的实际路径 !!!

# 3. 指定一个文件来存储授权令牌 (token)，这样下次运行不用重新授权
TOKEN_FILE = 'token.json'

# 4. 指定你要下载的 Google Drive 文件夹的 ID
#    如何获取文件夹 ID？打开 Google Drive 网页，进入那个文件夹，
#    看浏览器地址栏，类似 .../folders/THIS_IS_THE_FOLDER_ID
DRIVE_FOLDER_ID = '1MT28KK0uY9ff4YX8tAUZZAeYFgG4rHPV'

# 5. 指定本地保存文件的目录
LOCAL_DOWNLOAD_DIR = r"C:\Users\27268\Desktop\item\sentinel2" # !!! 替换成你的本地路径 !!!
PERFORM_DELETE = True # !!! 设置为 False 则只下载不删除，用于测试 !!!
# --- 配置区结束 ---

def authenticate():
    """处理用户认证流程"""
    creds = None
    # token.json 文件存储了用户的访问和刷新令牌，在首次授权后创建。
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    # 如果没有有效凭据，让用户登录。
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"无法刷新令牌: {e}")
                # 如果刷新失败，删除旧 token 文件，强制重新授权
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)
                creds = None # 重置 creds
        # 如果还是没有 creds，启动授权流程
        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CLIENT_SECRET_FILE, SCOPES)
                # 使用 localhost 服务器进行授权流程
                # port=0 会自动选择一个可用端口
                creds = flow.run_local_server(port=0)
            except FileNotFoundError:
                 print(f"错误：找不到客户端密钥文件 '{CLIENT_SECRET_FILE}'。请确保路径正确并已下载该文件。")
                 return None
            except Exception as e:
                 print(f"授权过程中发生错误: {e}")
                 return None
        # 保存凭据供下次运行使用
        if creds:
             with open(TOKEN_FILE, 'w') as token:
                 token.write(creds.to_json())
    return creds

def download_file(service, file_id, file_name, local_dir):
    """下载单个文件，成功后根据标志决定是否删除"""
    local_file_path = os.path.join(local_dir, file_name)
    download_successful = False
    try:
        request = service.files().get_media(fileId=file_id)
        # 创建本地文件路径
        local_file_path = os.path.join(local_dir, file_name)
        # 确保目标目录存在
        os.makedirs(local_dir, exist_ok=True)

        print(f"正在下载 {file_name} 到 {local_file_path} ...")
        # 使用 io.BytesIO 作为内存缓冲区
        fh = io.BytesIO()
        # MediaIoBaseDownload 用于下载大文件，分块进行
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"下载进度: {int(status.progress() * 100)}%")

        # 下载完成后，将内存缓冲区的内容写入本地文件
        fh.seek(0) # 回到缓冲区开头
        with open(local_file_path, 'wb') as f:
            f.write(fh.read())
        print(f"文件 {file_name} 下载完成。")

    except HttpError as error:
        print(f'下载文件 {file_name} (ID: {file_id}) 时发生错误: {error}')
    except Exception as e:
        print(f'下载文件 {file_name} 时发生未知错误: {e}')


def download_and_delete_file(service, file_id, file_name, local_dir, perform_delete):
    """下载单个文件，成功后根据标志决定是否删除"""
    local_file_path = os.path.join(local_dir, file_name)
    download_successful = False
    try:
        request = service.files().get_media(fileId=file_id)
        os.makedirs(local_dir, exist_ok=True)
        print(f"正在下载 {file_name} 到 {local_file_path} ...")
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            if status: # 检查 status 是否为 None
                 print(f"下载进度: {int(status.progress() * 100)}%")

        fh.seek(0)
        with open(local_file_path, 'wb') as f:
            f.write(fh.read())
        print(f"文件 {file_name} 下载完成。")
        download_successful = True # 标记下载成功

    except HttpError as error:
        print(f'下载文件 {file_name} (ID: {file_id}) 时发生错误: {error}')
    except Exception as e:
        print(f'下载文件 {file_name} 时发生未知错误: {e}')

    # --- 添加删除逻辑 ---
    if download_successful and perform_delete:
        try:
            # time.sleep(1) # 可选：在删除前稍微等待一下，确保文件系统操作完成
            print(f"尝试删除 Google Drive 上的文件 {file_name} (ID: {file_id})...")
            service.files().delete(fileId=file_id).execute()
            print(f"成功删除 Google Drive 上的文件 {file_name}。")
        except HttpError as delete_error:
            print(f'!!! 删除文件 {file_name} (ID: {file_id}) 时发生错误: {delete_error}')
        except Exception as delete_e:
            print(f'!!! 删除文件 {file_name} 时发生未知错误: {delete_e}')
    elif download_successful and not perform_delete:
        print(f"已跳过删除文件 {file_name} (测试模式)。")
    # --- 删除逻辑结束 ---


def main():
    """主函数：认证、列出文件、下载并根据标志删除"""
    creds = authenticate()
    if not creds:
        print("无法完成认证，脚本退出。")
        return

    try:
        service = build('drive', 'v3', credentials=creds)
        print(f"\n正在查找文件夹 '{DRIVE_FOLDER_ID}' 中的文件...")
        query = f"'{DRIVE_FOLDER_ID}' in parents and mimeType != 'application/vnd.google-apps.folder'"
        page_token = None
        items_processed = 0
        while True:
            results = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name)',
                pageToken=page_token
            ).execute()
            items = results.get('files', [])
            if not items:
                if items_processed == 0 and not page_token:
                     print("在指定的文件夹中没有找到文件。")
                else:
                     print("\n已处理完文件夹中的所有文件。")
                break

            print(f"\n找到 {len(items)} 个文件在本页:")
            for item in items:
                file_id = item['id']
                file_name = item['name']
                print(f"--- 处理文件: {file_name} (ID: {file_id}) ---")
                # 调用包含删除逻辑的函数
                download_and_delete_file(service, file_id, file_name, LOCAL_DOWNLOAD_DIR, PERFORM_DELETE)
                items_processed += 1

            page_token = results.get('nextPageToken', None)
            if page_token is None:
                break

    except HttpError as error:
        print(f'调用 Drive API 时发生错误: {error}')
        # ... (可以保留之前的特定错误消息，如 403, 404) ...
    except Exception as e:
         print(f"发生未知错误: {e}")

if __name__ == '__main__':
    main()