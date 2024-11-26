import random
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import time
import re
from deep_translator import GoogleTranslator
import threading
import emoji
from typing import Optional, Tuple
import os
import requests
import base64
from datetime import datetime
import io
import codecs
import colorama
import psutil
import unicodedata
import json
from collections import deque
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import string
from collections import deque
import hashlib
from urllib.parse import quote_plus
from PIL import Image
import qrcode
from urllib.parse import urlparse
from gtts import gTTS  # Google Text-to-Speech
import glob


colorama.init()  # đây là hướng dẫn dành cho bot và bạn lam kiểu đéo gì cũng được nhưng phải đảm bảo đúng chính sách của nó
SYSTEM_INSTRUCTION = """
Bạn là Loki - người bạn thông minh và duyên dáng trong giao tiếp. Hãy tuân theo các nguyên tắc sau:

1. Phong cách trả lời:
   - Luôn thay đổi cách diễn đạt, tránh lặp lại khuôn mẫu
   - Sử dụng ngôn ngữ đời thường, như đang nhắn tin với bạn thân
   - Ưu tiên câu ngắn gọn (1-2 dòng) nhưng đầy đủ thông tin
   - Thêm chút hài hước nhẹ nhàng khi phù hợp 😊

2. Đa dạng hóa:
   - Linh hoạt thay đổi giữa các kiểu câu (hỏi, kể, đề xuất)
   - Dùng từ ngữ phong phú nhưng dễ hiểu
   - Thay đổi cách mở đầu và kết thúc câu
   - Kết hợp emoji sáng tạo để tăng sinh động 🌟

3. Nguyên tắc:
   - Trả lời súc tích là ưu tiên hàng đầu
   - Thẳng thắn, chân thành, không vòng vo
   - Thừa nhận giới hạn khi cần thiết
   - Giữ giọng điệu vui vẻ, thân thiện

Mục tiêu: Tạo trải nghiệm trò chuyện tự nhiên, thú vị và hiệu quả. 💫"""

# prompt dự phòng https://gist.github.com/tanbaycu/66a9a08a30b5eb3f7f7912499780af97/raw


def success_color(string):
    return f"{colorama.Fore.GREEN}{colorama.Style.BRIGHT}{string}{colorama.Style.RESET_ALL}"


def error_color(string):
    return (
        f"{colorama.Fore.RED}{colorama.Style.BRIGHT}{string}{colorama.Style.RESET_ALL}"
    )


def warning_color(string):
    return f"{colorama.Fore.YELLOW}{colorama.Style.BRIGHT}{string}{colorama.Style.RESET_ALL}"


def info_color(string):
    return (
        f"{colorama.Fore.CYAN}{colorama.Style.BRIGHT}{string}{colorama.Style.RESET_ALL}"
    )


def debug_color(string):
    return f"{colorama.Fore.MAGENTA}{colorama.Style.BRIGHT}{string}{colorama.Style.RESET_ALL}"


def highlight_color(string):
    return f"{colorama.Back.BLUE}{colorama.Fore.WHITE}{colorama.Style.BRIGHT}{string}{colorama.Style.RESET_ALL}"


def system_color(string):
    return f"{colorama.Fore.BLUE}{colorama.Style.BRIGHT}{string}{colorama.Style.RESET_ALL}"  # thêm tí màu ngựa ngựa


def waiting_ui(timeout=5, content=""):
    for i in range(timeout):
        print(f"\r{warning_color(f'[{i+1}]')} -> {info_color(content)}", end="")
        time.sleep(1)
    print()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_error_code(error_type):

    prefix = {
        "Lỗi Tải Ảnh": "IMG",
        "Lỗi Xử Lý Ảnh": "PRC",
        "Lỗi API": "API",
        "Lỗi Hệ Thống": "SYS",
    }.get(
        error_type, "GEN"
    )  # gen cho lỗi chung

    random_part = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

    timestamp = int(time.time()) % 10000

    return f"{prefix}-{random_part}-{timestamp:04d}"


def format_error_message(error_type, error_details):
    error_code = generate_error_code(error_type)
    vnscii_chars = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯư"
    vnscii_suffix = "".join(random.choices(vnscii_chars, k=2))

    formatted_message = f"""
    ╔══════════════════════════════════════════════════════════════════════════════
    ║ Mã Lỗi: {error_code}{vnscii_suffix}
    ║ Loại Lỗi: {error_type}
    ║ Chi Tiết: {error_details}
    ║ Thời Gian: {time.strftime("%Y-%m-%d %H:%M:%S")}
    ╚══════════════════════════════════════════════════════════════════════════════
    """
    return formatted_message


def send_to_gemini(
    message,
    image_path=None,
    conversation_history=None,
    model="gemini-1.5-pro-latest",
    temperature=1,
    max_output_tokens=4096,
    top_p=0.95,
    top_k=1,
    max_retries=3,
    initial_delay=1,
):
    api_keys = ["YOUR_GEMINI_API_KEY_1", "YOUR_GEMINI_API_KEY_2"]

    for api_key in api_keys:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"  # base url
        headers = {"Content-Type": "application/json"}

        content_parts = []

        if conversation_history:  # sử dụng bộ nhớ
            for entry in conversation_history:
                content_parts.append({"text": f"{entry['role']}: {entry['content']}"})

        content_parts.append({"text": SYSTEM_INSTRUCTION + "\n\n" + message})

        if image_path:  # xử lý hình ảnh
            try:
                image_data = encode_image(image_path)
                content_parts.append(
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_data}}
                )
            except FileNotFoundError:
                return format_error_message(
                    "Lỗi Tải Ảnh", "Không tìm thấy file ảnh tại đường dẫn đã chỉ định."
                )
            except Exception as e:
                return format_error_message("Lỗi Xử Lý Ảnh", str(e))

        data = {
            "contents": [{"parts": content_parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "topP": top_p,
                "topK": top_k,
            },
        }  # body gửi về

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()

                response_data = response.json()
                generated_text = response_data["candidates"][0]["content"]["parts"][0][
                    "text"
                ]
                return generated_text
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    delay = (2**attempt + random.uniform(0, 1)) * initial_delay
                    print(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    return format_error_message("Lỗi API", f"Lỗi HTTP: {e}")
            except requests.exceptions.RequestException as e:
                return format_error_message("Lỗi API", f"Lỗi yêu cầu: {e}")
            except (KeyError, IndexError) as e:
                return format_error_message(
                    "Lỗi Hệ Thống", f"Lỗi phân tích phản hồi: {e}"
                )
            except Exception as e:
                return format_error_message("Lỗi Hệ Thống", f"Lỗi không xác định: {e}")

    return format_error_message(
        "Lỗi API", "Không thể kết nối với API Gemini sau nhiều lần thử."
    )  # bao hàm lỗi


def generate_image_huggingface(prompt):
    API_URL = (
        "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    )
    headers = {"Authorization": "Bearer YOUR_HUGGINGFACE_TOKEN"}

    payload = {
        "inputs": prompt,
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    image_bytes = response.content

    # Lưu hình ảnh
    with open("generated_image.jpg", "wb") as file:
        file.write(image_bytes)

    return "generated_image.jpg"


""" sử dụng thư viện thay vì sử dụng api => nhưng nó chậm vc
def send_to_gemini(
    message,
    image_path=None,
    conversation_history=None,
    model="gemini-1.5-pro-002",
    temperature=1,
    max_output_tokens=8162,
    top_p=0.95,
    top_k=40,
    max_retries=3,
    retry_delay=5
):
    api_key = ""
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name=model)

    content_parts = []

    if conversation_history:
        for entry in conversation_history:
            content_parts.append({"role": entry['role'], "parts": [entry['content']]})

    content_parts.append({"role": "user", "parts": [SYSTEM_INSTRUCTION + "\n\n" + message]})

    if image_path:
        image = Image.open(image_path)
        content_parts[-1]["parts"].append(image)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                content_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_p=top_p,
                    top_k=top_k
                )
            )

            generated_text = response.text
            return generated_text
        except genai.types.ResponseError as e:
            if "Resource has been exhausted" in str(e):
                print(warning_color(f"API quota exhausted. Attempt {attempt + 1} of {max_retries}. Retrying in {retry_delay} seconds..."))
                time.sleep(retry_delay)
            else:
                print(error_color(f"Lỗi khi kết nối với API Gemini: {e}"))
                break
        except Exception as e:
            print(error_color(f"Lỗi không xác định khi kết nối với API Gemini: {e}"))
            break

    # If all retries fail or another error occurs, use fallback response
    return generate_fallback_response(message)

def generate_fallback_response(message):
    fallback_responses = [
        "Xin lỗi, hiện tại tôi đang gặp khó khăn trong việc xử lý yêu cầu của bạn. Bạn có thể thử lại sau không?",
        "Rất tiếc, tôi không thể trả lời câu hỏi của bạn lúc này. Hãy thử lại sau vài phút nữa nhé.",
        "Có vẻ như hệ thống đang bận. Bạn có thể đặt câu hỏi khác hoặc chờ một lát rồi thử lại.",
        "Tôi đang gặp sự cố kỹ thuật. Xin lỗi vì sự bất tiện này. Hãy thử lại sau nhé!",
        "Hệ thống đang quá tải. Tôi sẽ cố gắng trả lời câu hỏi của bạn sớm nhất có thể."
    ]
    return random.choice(fallback_responses)
"""


class FacebookLogin:
    def __init__(self, email_or_phone, password):
        self.email_or_phone = email_or_phone
        self.password = password

    def login_twice(self):

        try:

            print(
                info_color("[*] Đang thực hiện đăng nhập lần 1...")
            )  # lấy cookies lần đầu cho facebook khỏi xác thực
            self.driver.get("https://www.messenger.com/login/")

            email_field = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.ID, "email"))
            )
            email_field.send_keys(self.email_or_phone)
            print(info_color("[*] Đã nhập email/phone"))

            time.sleep(0.2)
            password_field = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.ID, "pass"))
            )
            password_field.send_keys(self.password)
            print(info_color("[*] Đã nhập password"))

            time.sleep(0.2)
            login_button = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.ID, "loginbutton"))
            )
            login_button.click()
            print(success_color("[*] Đã click nút đăng nhập"))

            print(info_color("[*] Đợi 30 giây trước khi đăng nhập lại..."))
            time.sleep(30)

            print(info_color("[*] Đang quay lại trang đăng nhập..."))
            self.driver.back()

            print(info_color("[*] Đang thực hiện đăng nhập lần 2..."))
            password_field = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.ID, "pass"))
            )
            password_field.send_keys(self.password)
            print(info_color("[*] Đã nhập lại password"))

            login_button = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.ID, "loginbutton"))
            )
            login_button.click()
            print(success_color("[*] Đã click nút đăng nhập lần 2"))

            print(success_color("[+] Đăng nhập thành công!"))
            return True

        except Exception as e:
            print(
                error_color(f"[!] Lỗi trong quá trình đăng nhập: {str(e)}")
            )  # facebook đòi xác thực với anh à
            return False


class LoginCreateSession(FacebookLogin):  # đăng nhập
    def __init__(self, email_or_phone, password, group_or_chat):
        super().__init__(email_or_phone, password)
        options = webdriver.ChromeOptions()

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--log-level=3")
        options.add_argument("--headless=old")  # đừng bật cái này nhe

        self.browser = webdriver.Chrome(options=options, keep_alive=True)
        self.driver = self.browser  # Để FacebookLogin có thể sử dụng

        self.group_or_chat = group_or_chat

        # Thực hiện đăng nhập 2 lần
        self.login_twice()
        self.check_verify()
        self.pass_notify()
        self.to_group_or_chat()

    def get_to_mes(self):  # truy cập vào messenger bằng trình duyệt headless
        self.browser.get("https://www.messenger.com/login/")
        print(info_color("[*] Navigating to Messenger login page"))

    def login(self):
        time.sleep(1)
        try:
            WebDriverWait(self.browser, 30).until(
                EC.presence_of_element_located((By.ID, "email"))
            ).send_keys(self.email_or_phone)
            print(info_color("[*] Entered email/phone"))
        except:
            print(error_color("[!] Failed to enter email/phone"))
            raise

        time.sleep(0.2)
        try:
            WebDriverWait(self.browser, 30).until(
                EC.presence_of_element_located((By.ID, "pass"))
            ).send_keys(self.password)
            print(info_color("[*] Entered password"))
        except:
            print(error_color("[!] Failed to enter password"))
            raise

        time.sleep(0.2)
        try:
            WebDriverWait(self.browser, 30).until(
                EC.presence_of_element_located((By.ID, "loginbutton"))
            ).click()
            print(success_color("[*] Clicked login button"))
        except:
            print(error_color("[!] Failed to click login button"))
            raise

    def check_verify(self):
        waiting_ui(5, "Please wait for 5 seconds")
        input(
            warning_color(
                "[!] Please verify access (if required) and press Enter to continue\n-> "
            )
        )
        self.browser.get("https://www.messenger.com/login/")
        print(info_color("[*] Checking for verification requests..."))
        try:
            continue_with_acc = WebDriverWait(self.browser, 2.5).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//span[contains(@class, '_2hyt')]")
                )
            )
            continue_with_acc.click()
            print(success_color("[*] Verified and continuing..."))
        except Exception as e:
            print(info_color("[*] No verification requests found, continuing..."))

    def pass_notify(self):
        print(info_color("[*] Checking for message sync requests..."))
        try:
            x1 = WebDriverWait(self.browser, 2.5).until(
                EC.presence_of_element_located((By.XPATH, "//div[@aria-label='Đóng']"))
            )
            x1.click()
            x2 = WebDriverWait(self.browser, 2.5).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//span[contains(@class, 'x1lliihq') and text()='Không đồng bộ']",
                    )
                )
            )
            x2.click()
            print(success_color("[*] Messages synced successfully!"))
        except Exception as e:
            print(info_color("[*] No message sync requests found"))

    def to_group_or_chat(self):
        print(info_color("[*] Navigating to chat box..."))
        self.browser.get(self.group_or_chat)


class Listener(LoginCreateSession):  # lắng nghe tin nhắn từ đoạn chat và người dùng
    def __init__(self, email_or_phone, password, group_or_chat):
        super().__init__(email_or_phone, password, group_or_chat)

        self.his_inp = ""
        self.current_inp = ""
        self.his_img_value = ""
        self.current_img_value = ""
        self.current_image = None
        self.waiting = True
        self.username = None
        self.sending_img = False

        threading.Thread(target=self.listening).start()
        self.waiting_setting_up()

    def remove_emoji(self, text):
        return emoji.replace_emoji(text, replace="")

    def first_input_message_init(self):
        print(info_color("[*] Initializing messages and images input..."))
        try:
            message = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        '(//div[@class="html-div xexx8yu x4uap5 x18d9i69 xkhd6sd x1gslohp x11i5rnm x12nagc x1mh8g0r x1yc453h x126k92a x18lvrbx"])[last()]',
                    )
                )
            )
            message = message.text

            self.his_inp = message
            self.current_inp = message

            print(info_color("[*] Searching for images..."))
            img = None
            try:
                img = WebDriverWait(self.browser, 2).until(
                    EC.presence_of_all_elements_located(
                        (
                            By.XPATH,
                            "//img[@alt='Open photo' and contains(@class, 'xz74otr xmz0i5r x193iq5w')]",
                        )
                    )
                )
                print(success_color("[*] Image found!"))
            except Exception as e:
                print(warning_color("[!] No image found in the chat box"))

            self.his_img_value = str(img)[-100:]
            self.current_img_value = str(img)[-100:]
        except Exception as e:
            print(error_color(f"[!] Error initializing messages and images input: {e}"))

    def check_new_message(self):
        try:
            message = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        '(//div[@class="html-div xexx8yu x4uap5 x18d9i69 xkhd6sd x1gslohp x11i5rnm x12nagc x1mh8g0r x1yc453h x126k92a x18lvrbx"])[last()]',
                    )
                )
            )
            message = message.text

            if message != self.his_inp:
                print(info_color("[*] Searching for username..."))
                username = None
                try:
                    username = WebDriverWait(self.browser, 2.5).until(
                        EC.presence_of_all_elements_located(
                            (
                                By.XPATH,
                                "//span[contains(@class, 'html-span') and contains(@class, 'xdj266r') and contains(@class, 'x11i5rnm') and contains(@class, 'xat24cr') and contains(@class, 'x1mh8g0r') and contains(@class, 'xexx8yu') and contains(@class, 'x4uap5') and contains(@class, 'x18d9i69') and contains(@class, 'xkhd6sd') and contains(@class, 'x1hl2dhg') and contains(@class, 'x16tdsg8') and contains(@class, 'x1vvkbs')]",
                            )
                        )
                    )
                except Exception as e:
                    print(warning_color("[!] Error finding username"))
                if username is not None:
                    print(success_color(f"[*] Found username -> {username[-1].text}"))
                    self.username = username[-1].text
                self.current_inp = message
        except Exception as e:
            print(error_color(f"[!] Error checking for new messages: {e}"))

    def check_new_image(self):
        print(system_color("[!] finding image..."))
        img = None
        try:
            try:
                img = WebDriverWait(self.browser, 2).until(
                    EC.presence_of_all_elements_located(
                        (
                            By.XPATH,
                            "//img[@alt='Open photo' and contains(@class, 'xz74otr xmz0i5r x193iq5w')]",
                        )
                    )
                )
            except:
                img = WebDriverWait(self.browser, 2).until(
                    EC.presence_of_all_elements_located(
                        (
                            By.XPATH,
                            "//img[@alt='Mở ảnh' and contains(@class, 'xz74otr xmz0i5r x193iq5w')]",
                        )
                    )
                )
            print(success_color("[*] founded an image!"))
        except Exception as e:
            print(error_color("[!] error when finding image!"))
        if img is not None:
            try:
                img = img[-1].get_attribute("src")
            except:
                img = img[-1].get_attribute("src")
            if img[:4] == "http":
                img = requests.get(img).content
            else:
                img = base64.b64decode(img[23:])
            if str(img)[-100:] == self.his_img_value:
                pass
            else:
                if self.sending_img:
                    self.current_img_value = str(img)[-100:]
                    self.his_img_value = str(img)[-100:]
                    self.sending_img = False
                else:
                    self.current_image = img
                    self.current_img_value = str(img)[-100:]

    def listening(self):
        print(highlight_color("[#] Listening for new messages and images..."))
        self.first_input_message_init()
        while True:
            try:
                time.sleep(0.5)
                self.check_new_message()
                self.check_new_image()

                if self.waiting:
                    self.waiting = False

            except Exception as e:
                print(error_color(f"[!] Error during listening: {e}"))
                continue

    def waiting_setting_up(self):
        time.sleep(1)
        print(info_color("[*] Waiting for setup to complete..."))
        while not self.waiting:
            pass
        print(success_color("[*] Setup completed successfully!"))


class Sender(Listener):  # class để gửi ảnh và tin nhắn
    def __init__(self, email_or_phone, password, group_or_chat):
        super().__init__(email_or_phone, password, group_or_chat)

    def send_message(self, inp=None, inp_down_line=None):
        try:
            try:
                send_msg = WebDriverWait(self.browser, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//p[@class='xat24cr xdj266r']")
                    )
                )
            except:
                send_msg = WebDriverWait(self.browser, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//p[@class='xat24cr xdj266r']")
                    )
                )

            """try:
                send_msg = WebDriverWait(self.browser, 10).until(EC.presence_of_element_located((By.XPATH, "//p[@class='xat24cr xdj266r']")))
            except:
                send_msg = WebDriverWait(self.browser, 10).until(EC.presence_of_element_located((By.XPATH, "//p[@class='xat24cr xdj266r']")))"""

            if inp is not None:
                inp = " ".join(inp.split())
                # Convert emoji shortcodes to Unicode
                inp_with_emojis = emoji.emojize(inp, language="alias")

                # Use ActionChains for more reliable input
                actions = ActionChains(self.browser)
                actions.move_to_element(send_msg)
                actions.click()
                actions.send_keys(inp_with_emojis)
                actions.send_keys(Keys.ENTER)
                actions.perform()

            elif inp_down_line is not None:
                actions = ActionChains(self.browser)
                actions.move_to_element(send_msg)
                actions.click()

                for inp in inp_down_line:
                    inp_with_emojis = emoji.emojize(inp, language="alias")
                    actions.send_keys(inp_with_emojis)
                    actions.key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(
                        Keys.SHIFT
                    )

                actions.send_keys(Keys.ENTER)
                actions.perform()

            elif inp_down_line is not None:
                for inp in inp_down_line:
                    try:
                        send_msg.send_keys(self.remove_emoji(inp + " "))
                    except:
                        send_msg.send_keys(self.remove_emoji(inp + " "))
                    try:
                        send_msg.send_keys(Keys.SHIFT, Keys.ENTER)
                    except:
                        send_msg.send_keys(Keys.SHIFT, Keys.ENTER)
                    time.sleep(0.2)
                try:
                    send_msg.send_keys(Keys.ENTER)
                except:
                    send_msg.send_keys(Keys.ENTER)

                print(success_color("[*] send message is successed!"))

        except Exception as e:
            print(error_color("[!] error when send messgae!"))
        self.his_inp = self.current_inp

    def send_image(
        self, img_path: str = "./image_model/output_image.png", message=None
    ):
        self.sending_img = True
        try:
            upload_image = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//input[@type="file" and contains(@class, "x1s85apg")]')
                )
            )
        except:
            upload_image = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//input[@type="file" and contains(@class, "x1s85apg")]')
                )
            )
        time.sleep(0.2)
        try:
            upload_image.send_keys(os.path.abspath(img_path))
        except:
            upload_image.send_keys(os.path.abspath(img_path))
        time.sleep(0.2)
        try:
            send_msg = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//p[@class='xat24cr xdj266r']")
                )
            )
        except:
            send_msg = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//p[@class='xat24cr xdj266r']")
                )
            )
        time.sleep(0.2)
        try:
            send_msg.send_keys(" " if message == None else message)
        except:
            send_msg.send_keys(" " if message == None else message)
        time.sleep(0.2)
        try:
            send_msg.send_keys(Keys.ENTER)
        except:
            send_msg.send_keys(Keys.ENTER)


#
#


class CodeAssistant:
    def __init__(self):
        self._github_token = ""
        self._gemini_api_key = ""
        self._supported_languages = {
            "python": ".py",
            "javascript": ".js",
            "java": ".java",
            "cpp": ".cpp",
            "csharp": ".cs",
            "ruby": ".rb",
            "go": ".go",
            "rust": ".rs",
            "typescript": ".ts",
            "swift": ".swift",
            "kotlin": ".kt",
            "php": ".php",
            "html": ".html",
            "css": ".css",
            "sql": ".sql",
        }
        self._setup_gemini()

    def _setup_gemini(self):
        genai.configure(api_key=self._gemini_api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")

    def _generate_code_and_explanation(
        self, language: str, request: str
    ) -> Tuple[str, str]:
        prompt = self._create_advanced_prompt(language, request)

        try:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    top_p=1,
                    top_k=1,
                    max_output_tokens=8192,
                ),
            )

            full_response = response.text
            code_start = full_response.find(f"```{language}")
            code_end = full_response.find("```", code_start + len(language) + 3)
            explanation_start = full_response.find("EXPLANATION:")

            if code_start != -1 and code_end != -1 and explanation_start != -1:
                code = full_response[code_start + len(language) + 3 : code_end].strip()
                explanation = full_response[
                    explanation_start + len("EXPLANATION:") :
                ].strip()
            else:
                raise ValueError("Failed to extract code or explanation properly")

            return code, explanation
        except Exception as e:
            return f"# Error generating code: {str(e)}", f"Error: {str(e)}"

    def _create_advanced_prompt(self, language: str, request: str) -> str:
        return f"""
        As an expert {language} developer with years of experience, your task is to generate highly optimized, production-ready {language} code for the following request:

        {request}

        Your response MUST adhere to the following strict guidelines:

        1. CODE IMPLEMENTATION:
           - Provide a COMPLETE, FULLY FUNCTIONAL implementation that addresses ALL aspects of the request.
           - The code MUST be syntactically correct and follow the latest best practices and conventions for {language}.
           - Include ALL necessary imports, function definitions, classes, and main execution blocks.
           - Implement proper error handling and edge case management.
           - Ensure the code is optimized for performance and memory efficiency.
           - Use appropriate data structures and algorithms to solve the problem efficiently.
           - Follow SOLID principles and implement clean, maintainable code.
           - Add clear, concise comments to explain complex logic or algorithms.
           - Implement proper input validation and data sanitization where necessary.
           - Use meaningful variable and function names that clearly convey their purpose.
           - Adhere to the language-specific style guide (e.g., PEP 8 for Python, Google Style Guide for Java).

        2. COMPREHENSIVE EXPLANATION:
           - Provide an in-depth, technical explanation of the code in English.
           - Cover the following aspects in detail:
             a) Overall architecture and design patterns used in the solution
             b) Detailed explanation of each major component, class, or function
             c) Time and space complexity analysis of key algorithms
             d) Justification for chosen data structures and their impact on performance
             e) Explanation of any advanced language features or libraries used
             f) Discussion of potential edge cases and how they are handled
             g) Scalability considerations and potential optimizations for larger datasets
             h) Any important design decisions or trade-offs made, with reasoning
             i) Suggestions for further improvements or alternative approaches
           - The explanation should be clear, concise, and technically accurate, suitable for an experienced developer audience.

        3. TESTING AND QUALITY ASSURANCE:
           - Outline a testing strategy for the implemented solution.
           - Provide example test cases covering various scenarios, including edge cases.
           - Discuss potential integration and performance testing approaches.

        Format your response EXACTLY as follows:

        CODE:
        ```{language}
        [Your complete, production-ready code here]
        ```

        EXPLANATION:
        [Your comprehensive, technical explanation here]

        TESTING:
        [Outline of testing strategy and example test cases]

        Failure to follow these guidelines precisely will result in an incorrect and unusable response. Ensure your code and explanation demonstrate the highest level of expertise and attention to detail.
        """

    def _create_gist(self, language: str, content: str) -> Optional[str]:
        url = "https://api.github.com/gists"
        headers = {
            "Authorization": f"token {self._github_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        file_extension = self._supported_languages.get(language, ".txt")
        data = {
            "description": f"Optimized {language.capitalize()} Solution",
            "public": False,
            "files": {f"solution{file_extension}": {"content": content}},
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["html_url"]
        except requests.exceptions.RequestException as e:
            print(f"Error creating Gist: {str(e)}")
            return None

    def process_code_request(self, message: str) -> str:
        parts = message.split(maxsplit=2)
        if len(parts) < 3:
            return "Cách sử dụng: /code [ngôn ngữ] [yêu cầu chi tiết]"

        language = parts[1].lower()
        request = parts[2]

        if language not in self._supported_languages:
            return f"Ngôn ngữ không được hỗ trợ. Các ngôn ngữ được hỗ trợ là: {', '.join(self._supported_languages.keys())}"

        generated_code, explanation = self._generate_code_and_explanation(
            language, request
        )
        gist_url = self._create_gist(language, generated_code)

        translated_explanation = self.translate(explanation)

        if gist_url:
            response = f"Đã tạo giải pháp tối ưu thành công! Xem toàn bộ mã nguồn tại đây: {gist_url}\n\nGiải thích chi tiết:\n{translated_explanation}"
        else:
            response = f"Không thể tạo Gist. Đây là giải thích chi tiết cho mã nguồn được tạo ra:\n\n{translated_explanation}"

        return response

    def translate(self, text: str) -> str:
        translator = GoogleTranslator(source="en", target="vi")
        max_length = 5000
        parts = [text[i : i + max_length] for i in range(0, len(text), max_length)]
        translated_parts = [translator.translate(part) for part in parts]
        return " ".join(translated_parts)


class WordChainGame:
    def __init__(self):
        self.used_words = set()
        self.is_active = False
        self.current_word = None

        self.ai_prompt = """[SYSTEM: Bạn là trí tuệ nhân tạo chuyên về ngôn ngữ và trò chơi nối từ tiếng Việt. Nhiệm vụ của bạn là:

1. KIỂM TRA TÍNH HỢP LỆ CỦA TỪ NGƯỜI CHƠI:
- Phân tích xem cụm từ có phải là từ ghép có nghĩa trong tiếng Việt không
- Kiểm tra ngữ nghĩa và cách dùng phổ biến
- Đảm bảo không phải từ lóng, tiếng địa phương hoặc biệt ngữ
- Xác định tính logic và mối liên kết giữa các từ trong cụm

2. QUY TẮC NỐI TỪ THÔNG MINH:
- Lấy từ cuối của người chơi làm từ đầu trong cụm từ mới
- Tạo từ ghép có nghĩa rõ ràng, phổ biến trong tiếng Việt
- Ưu tiên các từ ghép:
  + Có tính kết nối cao (dễ nối tiếp)
  + Thuộc nhiều lĩnh vực khác nhau (đa dạng chủ đề)
  + Tạo sự thú vị và thách thức cho người chơi

3. PHÂN TÍCH VÀ PHẢN HỒI:
Nếu từ người chơi không hợp lệ, trả về: "INVALID: lý do"
Nếu từ hợp lệ, trả về từ ghép mới theo format: "VALID: từ_ghép_mới"

VÍ DỤ CHUẨN:
Người: "con người" (hợp lệ) -> "VALID: người dân"
Người: "dân tình" (hợp lệ) -> "VALID: tình cảm"
Người: "cảm xúc" (hợp lệ) -> "VALID: xúc động"
Người: "con ngừi" (không hợp lệ) -> "INVALID: từ viết sai chính tả"
Người: "xyz abc" (không hợp lệ) -> "INVALID: không phải từ ghép có nghĩa"

4. KIỂM TRA TRÙNG LẶP:
- Từ đã sử dụng: {used_words}
- KHÔNG được dùng lại bất kỳ từ nào trong danh sách

Từ hiện tại cần xử lý: "{current_word}"

OUTPUT FORMAT:
- Nếu hợp lệ: VALID: từ_ghép_mới
- Nếu không hợp lệ: INVALID: lý_do]"""

    def make_move(self, word, gemini_response):
        word = word.lower().strip()

        # Kiểm tra nếu người dùng đầu hàng
        if word in ["đầu hàng", "chịu thua", "thua"]:
            # Tạo từ mới cho lượt chơi mới
            new_word = self.generate_new_word(gemini_response)
            self.is_active = True
            self.used_words.clear()
            self.current_word = new_word
            return (
                f"🎯 Kết thúc lượt chơi trước!\n🎮 Bắt đầu lượt mới với từ: {new_word}"
            )

        # Kiểm tra yêu cầu giải thích
        if word.startswith("giải thích "):
            word_to_explain = word.replace("giải thích ", "", 1).strip()
            if (
                word_to_explain in self.used_words
                or word_to_explain == self.current_word
            ):
                explanation_prompt = f"Giải thích ngắn gọn ý nghĩa và cách dùng của từ '{word_to_explain}' trong 2-3 câu"
                explanation = gemini_response(explanation_prompt).strip()
                return f"💡 {explanation}\n\nTừ cuối để nối tiếp: '{self.current_word.split()[-1]}'"
            else:
                return (
                    f"❌ Từ '{word_to_explain}' chưa được sử dụng trong lượt chơi này"
                )

        # Xử lý nối từ bình thường
        if not self.is_active:
            self.is_active = True
            self.used_words.clear()

            prompt = self.ai_prompt.format(
                current_word=word, used_words=list(self.used_words)
            )
            response = gemini_response(prompt).strip()

            if response.startswith("INVALID:"):
                self.is_active = False
                return f"❌ {response.replace('INVALID:', 'Từ không hợp lệ:')}\n💡 Vui lòng /noitu + từ ghép có nghĩa khác"

            bot_word = response.replace("VALID:", "").strip()
            self.used_words.add(word)
            self.used_words.add(bot_word)
            self.current_word = bot_word
            return f"🎮 Bắt đầu nối từ!\nBạn: {word}\n🤖 Bot: {bot_word}"

        if word in self.used_words:
            return f"❌ '{word}' đã được sử dụng!\n📝 Các từ đã dùng: {', '.join(self.used_words)}\n🏆 Bot thắng!\n\nDùng /noitu + từ mới để chơi lại!"

        last_word = self.current_word.split()[-1]
        if not word.startswith(last_word):
            return f"❌ Từ mới phải bắt đầu bằng '{last_word}'\n💡 Từ trước đó: {self.current_word}"

        prompt = self.ai_prompt.format(
            current_word=word, used_words=list(self.used_words)
        )

        response = gemini_response(prompt).strip()

        if response.startswith("INVALID:"):
            return f"❌ {response.replace('INVALID:', 'Từ không hợp lệ:')}\n💡 Hãy dùng từ ghép có nghĩa khác"

        bot_word = response.replace("VALID:", "").strip()
        if not bot_word or bot_word in self.used_words:
            return f"🎉 Bạn thắng! Bot không tìm được từ phù hợp.\n📝 Các từ đã dùng: {', '.join(self.used_words)}\n\nDùng /noitu + từ mới để chơi lại!"

        self.used_words.add(word)
        self.used_words.add(bot_word)
        self.current_word = bot_word
        return f"🤖 Bot: {bot_word}"

    def generate_new_word(self, gemini_response):
        """Tạo từ ghép mới ngẫu nhiên cho lượt chơi mới"""
        prompt = """Hãy tạo một từ ghép tiếng Việt có nghĩa và phổ biến (ví dụ: con người, nhà cửa, học sinh).

        Yêu cầu:
        - Chỉ trả về từ ghép dạng text thuần, không kèm biểu tượng cảm xúc hay ký tự đặc biệt
        - Không thêm dấu câu hay định dạng
        - Không giải thích hay bổ sung thông tin
        - Chỉ trả về 2 từ đơn ghép lại thành từ ghép có nghĩa"""

        response = gemini_response(prompt).strip()
        return response.lower()


class URLTools:
    def __init__(self):
        self.history = []
        self.max_history = 50
        self.temp_dir = "temp_files"

        # Tạo thư mục temp nếu chưa có
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def create_short_url(self, long_url):
        """Rút gọn link sử dụng TinyURL"""
        try:
            # Kiểm tra URL hợp lệ
            result = urlparse(long_url)
            if not all([result.scheme, result.netloc]):
                return "❌ URL không hợp lệ! Hãy đảm bảo URL bắt đầu với http:// hoặc https://"

            # Gọi TinyURL API
            api_url = f"http://tinyurl.com/api-create.php?url={quote_plus(long_url)}"
            response = requests.get(api_url, timeout=10)

            if response.status_code == 200:
                short_url = response.text
                # Lưu vào lịch sử
                self.history.append(
                    {
                        "type": "short_url",
                        "original": long_url,
                        "shortened": short_url,
                        "timestamp": datetime.now(),
                    }
                )
                self._trim_history()

                return f"""🔗 Link gốc: {long_url}
✂️ Link rút gọn: {short_url}
📅 Thời gian tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

            return "❌ Không thể rút gọn link. Vui lòng thử lại sau!"

        except requests.Timeout:
            return "❌ Hết thời gian chờ. Vui lòng thử lại!"
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"

    def create_qr_code(self, content):
        """Tạo QR code"""
        try:
            # Tạo tên file độc nhất trong thư mục temp
            filename = os.path.join(
                self.temp_dir,
                f"qr_{hashlib.md5(content.encode()).hexdigest()[:10]}.png",
            )

            # Tạo QR với cấu hình tối ưu
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            qr.add_data(content)
            qr.make(fit=True)

            # Tạo và lưu ảnh QR
            qr_image = qr.make_image(fill_color="black", back_color="white")
            qr_image.save(filename)

            # Lưu lịch sử
            self.history.append(
                {
                    "type": "qr_code",
                    "content": content,
                    "filename": filename,
                    "timestamp": datetime.now(),
                }
            )
            self._trim_history()

            return filename

        except Exception as e:
            return f"❌ Lỗi tạo QR code: {str(e)}"

    def analyze_url(self, url):
        """Phân tích URL"""
        try:
            # Parse URL
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return "❌ URL không hợp lệ!"

            # Kiểm tra URL
            response = requests.head(url, allow_redirects=True, timeout=5)

            # Phân tích thông tin
            info = {
                "domain": parsed.netloc,
                "path": parsed.path or "/",
                "protocol": parsed.scheme,
                "status": response.status_code,
                "content_type": response.headers.get("content-type", "Không xác định"),
                "size": response.headers.get("content-length", "Không xác định"),
            }

            # Lưu lịch sử
            self.history.append(
                {
                    "type": "analysis",
                    "url": url,
                    "info": info,
                    "timestamp": datetime.now(),
                }
            )
            self._trim_history()

            return f"""🔍 Thông tin URL:

🌐 Tên miền: {info['domain']}
📁 Đường dẫn: {info['path']}
🔒 Giao thức: {info['protocol']}
📊 Trạng thái: {info['status']}
📝 Loại nội dung: {info['content_type']}
📦 Kích thước: {info['size']}
⏰ Thời gian kiểm tra: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        except requests.Timeout:
            return "❌ Hết thời gian chờ khi kiểm tra URL!"
        except requests.RequestException:
            return "❌ Không thể kết nối tới URL!"
        except Exception as e:
            return f"❌ Lỗi phân tích URL: {str(e)}"

    def get_history(self, limit=5):
        """Xem lịch sử thao tác"""
        try:
            if not self.history:
                return "📝 Chưa có lịch sử thao tác nào!"

            items = self.history[-limit:]

            result = f"📜 {limit} thao tác gần đây:\n\n"
            for item in reversed(items):
                result += f"⏰ {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                if item["type"] == "short_url":
                    result += f"🔗 Rút gọn: {item['original']} -> {item['shortened']}\n"
                elif item["type"] == "qr_code":
                    result += f"📱 QR Code: {item['content']}\n"
                elif item["type"] == "analysis":
                    result += f"🔍 Phân tích: {item['url']}\n"
                result += "---\n"

            return result

        except Exception as e:
            return f"❌ Lỗi lấy lịch sử: {str(e)}"

    def _trim_history(self):
        """Giới hạn kích thước lịch sử"""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def cleanup_temp_files(self):
        """Dọn dẹp file tạm"""
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"Lỗi dọn dẹp file tạm: {str(e)}")


class Voice:
    def __init__(self, bot):
        self.bot = bot
        self.is_active = False
        self.tts = gTTS
        self.temp_dir = "temp_voice"
        self.last_audio = None

        # Tạo thư mục temp nếu chưa tồn tại
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def make_move(self, message):
        """Xử lý tin nhắn và trả về phản hồi"""
        message = message.lower().strip()

        # Kiểm tra nếu người dùng muốn tắt voice
        if message in ["tắt voice", "tắt", "dừng", "/endvoice"]:
            self.is_active = False
            self.cleanup_temp_files()
            return "🔇 Đã tắt chế độ voice chat"

        # Khởi động voice mode
        if not self.is_active:
            if message == "start":
                self.is_active = True
                return "🎙️ Đã bật chế độ voice chat!\n💡 Gửi tin nhắn để nhận phản hồi bằng giọng nói\n❌ Gửi '/endvoice' để tắt"
            else:
                self.is_active = True
                return self._process_voice_message(message)

        # Xử lý tin nhắn trong voice mode
        return self._process_voice_message(message)

    def _process_voice_message(self, message):
        try:
            # Lấy phản hồi từ Gemini
            response = send_to_gemini(message)

            # Tạo file audio
            audio_path = self._text_to_speech(response)

            if audio_path:
                # Lưu đường dẫn audio cuối cùng
                if self.last_audio and os.path.exists(self.last_audio):
                    os.remove(self.last_audio)
                self.last_audio = audio_path

                # Gửi file audio qua bot
                self.bot.send_audio(audio_path)
                return None  # Không trả về text message
            else:
                return "❌ Không thể chuyển đổi văn bản thành giọng nói"

        except Exception as e:
            return f"❌ Lỗi xử lý voice: {str(e)}"

    def _text_to_speech(self, text):
        """Chuyển đổi văn bản thành file âm thanh"""
        try:
            # Tạo tên file duy nhất
            timestamp = int(time.time())
            audio_file = os.path.join(self.temp_dir, f"voice_{timestamp}.mp3")

            # Chuyển đổi text thành speech
            tts = self.tts(text=text, lang="vi")
            tts.save(audio_file)

            return audio_file

        except Exception as e:
            print(error_color(f"Lỗi chuyển đổi text to speech: {str(e)}"))
            return None

    def cleanup_temp_files(self):
        """Dọn dẹp các file tạm"""
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(warning_color(f"Không thể xóa file {file_path}: {str(e)}"))
        except Exception as e:
            print(error_color(f"Lỗi khi dọn dẹp thư mục temp: {str(e)}"))


class MesChat(Sender):  # hàm xử lý nâng cao
    def __init__(self, email_or_phone, password, group_or_chat):  # gemini_api_key
        super().__init__(email_or_phone, password, group_or_chat)
        self.image_counter = 0
        self.last_image_path = None
        self.is_running = True
        self.is_listening = True
        self.memory_summary = {}
        self.message_counter = 0
        self.MAX_MESSAGE_LENGTH = 2000
        self.current_language = "vi"
        self.conversation_history = []
        self.unsplash_access_key = ""
        self.github_token = ""
        self.max_history_length = 100
        self.longterm_memory_file = "longterm_memory.json"
        self.max_memory_size = 10000  # Giới hạn số lượng tin nhắn lưu trữ
        self.start_time = datetime.now()
        self.code_handler = CodeAssistant()
        self.fact_interval = 15  # thời gian cho mỗi fact
        self.fact_apis = [
            self.get_useless_fact,
            self.get_number_fact,
            self.get_cat_fact,
            self.get_dog_fact,
            self.get_today_in_history,
        ]
        self.used_facts = deque(maxlen=50)
        self.word_game = WordChainGame()  # siuu

    def new_message_listen(self):
        if self.his_inp != self.current_inp:
            self.his_inp = self.current_inp
            return self.current_inp
        else:
            return None

    def search_and_send_image(self, query: str):
        """Tìm kiếm và tạo gallery ảnh trên Telegra.ph với số lượng cố định 20 ảnh"""
        try:
            self.send_message(
                self.translate_message(f"🔍 Đang tìm ảnh cho '{query}'...")
            )

            # Request thêm ảnh để l���c (30 ảnh)
            encoded_query = quote_plus(query)

            try:
                unsplash_url = f"https://api.unsplash.com/photos/random?query={encoded_query}&count=30&client_id={self.unsplash_access_key}&orientation=landscape"
                response = requests.get(unsplash_url)
                response.raise_for_status()

                if not isinstance(response.json(), list):
                    raise ValueError("Invalid response format")

                # Lọc ảnh trùng lặp
                seen_urls = set()
                image_urls = []

                for photo in response.json():
                    if "urls" in photo and "regular" in photo["urls"]:
                        url = photo["urls"]["regular"]
                        if url not in seen_urls:
                            seen_urls.add(url)
                            image_urls.append(
                                {
                                    "url": url,
                                    "photographer": photo["user"]["name"],
                                    "likes": photo.get("likes", 0),
                                }
                            )

                if len(image_urls) < 20:
                    self.send_message(
                        self.translate_message(
                            f"❌ Không tìm đủ ảnh cho '{query}'. Vui lòng thử từ khóa khác."
                        )
                    )
                    return

                # Sắp xếp theo lượt thích và lấy 20 ảnh
                image_urls.sort(key=lambda x: x["likes"], reverse=True)
                image_urls = image_urls[:20]

                # Tạo gallery
                telegraph_content = [{"tag": "h3", "children": [f"Gallery: {query}"]}]

                for idx, img in enumerate(image_urls, 1):
                    telegraph_content.extend(
                        [
                            {
                                "tag": "figure",
                                "children": [
                                    {"tag": "img", "attrs": {"src": img["url"]}},
                                    {
                                        "tag": "figcaption",
                                        "children": [
                                            f"#{idx} - Photo by: {img['photographer']}"
                                        ],
                                    },
                                ],
                            }
                        ]
                    )

                # Upload lên Telegra.ph
                telegraph_response = requests.post(
                    "https://api.telegra.ph/createPage",
                    json={
                        "access_token": "",
                        "title": f"Gallery: {query}",
                        "author_name": "Loki Bot",
                        "content": telegraph_content,
                    },
                )

                if (
                    telegraph_response.status_code == 200
                    and telegraph_response.json().get("ok")
                ):
                    gallery_url = telegraph_response.json()["result"]["url"]
                    self.send_message(gallery_url)
                else:
                    self.send_message(
                        self.translate_message(
                            "❌ Không thể tạo gallery. Vui lòng thử lại sau."
                        )
                    )

            except Exception as e:
                print(error_color(f"[!] Lỗi Unsplash API: {str(e)}"))
                self.send_message(
                    self.translate_message(
                        "❌ Không thể tìm kiếm ảnh. Vui lòng thử lại sau."
                    )
                )

        except Exception as e:
            print(error_color(f"[!] Lỗi khi tạo gallery: {str(e)}"))
            self.send_message(
                self.translate_message("❌ Đã xảy ra lỗi. Vui lòng thử lại sau.")
            )

    def handle_image_command(self, message):
        """Xử lý lệnh /image với query có thể chứa nhiều từ"""
        try:
            parts = message.split(maxsplit=1)
            if len(parts) < 2:
                self.send_message(
                    self.translate_message(
                        "📝 Cú pháp: /image [chủ đề (1-4 từ)]\n"
                        "Ví dụ:\n"
                        "/image shiba\n"
                        "/image cute shiba inu\n"
                        "/image beautiful landscape nature photography"
                    )
                )
                return

            # Lấy query và kiểm tra số từ
            query = parts[1].strip()
            word_count = len(query.split())

            if not query:
                self.send_message(
                    self.translate_message("❌ Vui lòng nhập chủ đề cần tìm")
                )
                return

            if word_count > 4:
                self.send_message(
                    self.translate_message("❌ Chủ đề tìm kiếm không được quá 4 từ")
                )
                return

            # Gọi hàm search với query đã được xử lý
            self.search_and_send_image(query=query)

        except Exception as e:
            print(error_color(f"[!] Lỗi xử lý lệnh /image: {str(e)}"))
            self.send_message(
                self.translate_message("❌ Đã xảy ra lỗi. Vui lòng thử lại sau.")
            )

    def cleanup_image_directory(self, directory, max_images=50):
        """Giữ số lượng ảnh trong thư mục trong giới hạn cho phép"""
        try:
            # Lấy danh sách tất cả các file ảnh
            images = [
                f
                for f in os.listdir(directory)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
            ]

            # Sắp xếp theo thời gian tạo (mới nhất trước)
            images.sort(
                key=lambda x: os.path.getctime(os.path.join(directory, x)), reverse=True
            )

            # Xóa các ảnh cũ nếu vượt quá giới hạn
            if len(images) > max_images:
                for old_image in images[max_images:]:
                    try:
                        os.remove(os.path.join(directory, old_image))
                    except Exception as e:
                        print(
                            warning_color(
                                f"[!] Không thể xóa ảnh cũ {old_image}: {str(e)}"
                            )
                        )
        except Exception as e:
            print(error_color(f"[!] Lỗi khi dọn dẹp thư mục ảnh: {str(e)}"))

    def search_recipe(self, query):
        # Thêm cache để lưu các công thức đã tìm kiếm
        if not hasattr(self, "recipe_cache"):
            self.recipe_cache = {}

        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.recipe_cache:
            return self.recipe_cache[cache_key]

        api_key = ""  # Thay thế bằng API key của bạn
        base_url = "https://api.spoonacular.com/recipes/complexSearch"

        params = {
            "apiKey": api_key,
            "query": query,
            "number": 1,
            "addRecipeInformation": True,
            "fillIngredients": True,
            "instructionsRequired": True,
            "includeNutrition": False,
            "language": "en",  # Chỉ định tiếng Anh để có kết quả nhất quán
        }

        try:
            self.send_message(
                self.translate_message(f"Đang tìm kiếm công thức cho '{query}'...")
            )
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                self.send_message(
                    self.translate_message(
                        f"Xin lỗi, không tìm thấy công thức nào cho '{query}'. Vui lòng thử lại với từ khóa khác."
                    )
                )
                return

            recipe = data["results"][0]
            title = recipe.get("title", "Unknown Recipe")
            ingredients = recipe.get("extendedIngredients", [])
            instructions = recipe.get("analyzedInstructions", [{}])[0].get("steps", [])

            if not ingredients or not instructions:
                self.send_message(
                    self.translate_message(
                        f"Xin lỗi, không tìm thấy đủ thông tin cho công thức '{query}'."
                    )
                )
                return

            # Định dạng thông tin công thức
            message = self.translate_message(f"🍳 Công thức cho {title}:\n\n")
            message += self.translate_message("📋 Nguyên liệu:\n")
            for ingredient in ingredients:
                message += self.translate_message(f"• {ingredient['original']}\n")

            message += self.translate_message("\n👨‍🍳 Hướng dẫn:\n")
            for i, step in enumerate(instructions, 1):
                message += self.translate_message(f"{i}. {step['step']}\n")

            message += self.translate_message(
                f"\n🔗 Xem thêm chi tiết tại: {recipe['sourceUrl']}"
            )

            self.send_message(message)
        except requests.exceptions.RequestException as e:
            error_message = f"Có lỗi xảy ra khi tìm kiếm công thức: {str(e)}"
            self.send_message(self.translate_message(error_message))
        except Exception as e:
            error_message = f"Có lỗi không xác định xảy ra: {str(e)}"
            self.send_message(self.translate_message(error_message))
            print(f"Debug - Chi tiết lỗi: {e}")  # Thêm dòng này để gỡ lỗi

        self.recipe_cache[cache_key] = recipe_data
        return recipe_data

    def handle_recipe_command(self, message):
        parts = message.split(maxsplit=1)
        if len(parts) < 2:
            self.send_message(
                self.translate_message("Cách sử dụng: /recipe [tên món ăn]")
            )
            self.send_message(
                self.translate_message("Ví dụ: /recipe spaghetti carbonara")
            )
            return

        query = parts[1]
        self.search_recipe(query)

    def handle_new_message(self, message):
        print(system_color(f"Nhận được tin nhắn mới: {message}"))

        if self.word_game.is_active:
            if message.lower() == "/stopnoitu":
                self.word_game.is_active = False
                self.word_game.used_words.clear()
                self.send_message("🏁 Đã dừng trò chơi nối từ!")
                return

            # Nếu không phải lệnh dừng và game đang chạy, xử lý như tin nhắn nối từ
            if not message.startswith("/"):
                response = self.word_game.make_move(
                    message, lambda prompt: send_to_gemini(prompt)
                )
                self.send_message(response)
                return

        # Xử lý lệnh bắt đầu nối từ
        if message.lower().startswith("/noitu"):
            parts = message.split(maxsplit=1)
            if len(parts) > 1:
                response = self.word_game.make_move(
                    parts[1], lambda prompt: send_to_gemini(prompt)
                )
                self.send_message(response)
                return
            else:
                self.send_message("⚠️ Vui lòng nhập từ sau lệnh /noitu")
                return

        if hasattr(self, "voice_handler") and self.voice_handler.is_active:
            response = self.voice_handler.make_move(message)
            if response:  # Chỉ gửi text nếu có lỗi hoặc thông báo hệ thống
                self.send_message(response)
            return

        # Xử lý lệnh /voice
        if message.startswith("/voice"):
            if not hasattr(self, "voice_handler"):
                self.voice_handler = Voice(self)

            # Tách nội dung sau lệnh /voice nếu có
            parts = message.split(maxsplit=1)
            if len(parts) > 1:
                response = self.voice_handler.make_move(parts[1])
            else:
                response = self.voice_handler.make_move("start")

            if response:  # Chỉ gửi text nếu có phản hồi
                self.send_message(response)
            return

        # Thêm đoạn kiểm tra và tạo file nếu chưa tồn tại
        if not os.path.exists("conversation_history.json"):
            with open("conversation_history.json", "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print(info_color("[*] Đã tạo file conversation_history.json"))

        # Chỉ lưu tin nhắn không phải lệnh
        if message and message.strip() and not message.startswith("/"):
            self.conversation_history.append({"role": "user", "content": message})
            self.save_conversation_history()
            self.save_to_longterm_memory("user", message)
            print(info_color("[*] Đã lưu tin nhắn vào conversation_history.json"))

        if "/loki" in message.lower():
            self.send_random_dog_image()
            return

        elif message.lower().startswith("/code"):
            code_assistant = CodeAssistant()
            response = code_assistant.process_code_request(message)
            self.send_message(self.translate_message(response))
            return

        if message.lower().startswith("/language"):
            self.change_language(message)
            return

        elif message.startswith(("/short", "/qr", "/analyze", "/urlhistory")):
            self.handle_url_tools(message)
            return

        elif message.startswith("/image"):
            # Chuyển xử lý sang hàm handle_image_command
            self.handle_image_command(message)
            return
        elif message.lower().startswith("/summary"):
            try:

                time_range = "today"
                parts = message.split()
                if len(parts) > 1:
                    time_range = parts[1].lower()

                summary = self.summarize_conversations(time_range)
                self.send_message(summary)
            except Exception as e:
                error_msg = f"Lỗi khi đọc bộ nhớ dài hạn: {str(e)}"
                print(error_color(f"[!] {error_msg}"))
                self.send_message(self.translate_message(f"⚠️ {error_msg}"))
            return

        elif message.lower() == "/stop":
            self.is_running = False
            self.send_message(
                self.translate_message(
                    "Bot đã dừng hoạt động. Sử dụng lệnh /continue để tiếp tục."
                )
            )
            return
        elif message.lower() == "/continue":
            self.is_running = True
            self.send_message(self.translate_message("Bot đã tiếp tục hoạt động."))
            return
        elif message.lower() == "/save_history":
            self.save_conversation_history()
            self.send_message(self.translate_message("Đã lưu lịch sử trò chuyện."))
            return
        elif message.lower() == "/load_history":
            self.load_conversation_history()
            self.send_message(self.translate_message("Đã tải lịch sử trò chuyện."))
            return
        elif message.lower() == "/fact":
            self.fact_mode = True
            self.send_message(
                self.translate_message(
                    "Chế độ fact đã được kích hoạt. Sử dụng /stopfact để dừng."
                )
            )
            self.start_fact_loop()
            return

        elif message.lower() == "/stopfact":
            self.fact_mode = False
            self.send_message(self.translate_message("Chế độ fact đã được dừng."))
            return
        elif message.lower() == "/guide":
            self.send_guide()
            return

        elif message.lower().startswith("/recipe"):
            self.handle_recipe_command(message)
            return
        elif message.lower() == "/clear":
            try:
                self.conversation_history = []
                if os.path.exists("conversation_history.json"):
                    os.remove("conversation_history.json")
                self.send_message(
                    self.translate_message(
                        "Đã xóa lịch sử trò chuyện hiện tại. Bộ nhớ dài hạn vẫn được giữ nguyên."
                    )
                )
            except Exception as e:
                self.send_message(
                    self.translate_message(f"Lỗi khi xóa lịch sử: {str(e)}")
                )
            return
        elif message.lower().startswith("/createimage"):
            parts = message.split(maxsplit=1)
            if len(parts) < 2:
                self.send_message(
                    self.translate_message("Vui lòng cung cấp mô tả cho ảnh.")
                )
                return

            prompt = parts[1]
            image_path = generate_image_huggingface(prompt)
            self.send_image(image_path, "Đây là ảnh được tạo từ AI:")
            os.remove(image_path)

        elif message.lower() == "/help":
            self.send_help_message()
            return

        if not self.is_running:
            return

        self.supported_languages = {
            "vi": "Vietnamese",
            "en": "English",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
            "ja": "Japanese",
            "ko": "Korean",
            "zh-CN": "Chinese (Simplified)",
            "zh-TW": "Chinese (Traditional)",
            "th": "Thai",
            "id": "Indonesian",
            "ms": "Malay",
            "tr": "Turkish",
            "pl": "Polish",
            "sv": "Swedish",
            "da": "Danish",
            "fi": "Finnish",
        }

        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[
                -self.max_history_length :
            ]

        translated_message = self.translate_to_vietnamese(message)

        image_path = self.new_image_listen(return_file=True)
        if image_path:
            print(system_color(f"Nhận được hình ảnh mới: {image_path}"))
            self.last_image_path = image_path
            gemini_response = send_to_gemini(
                f"Hãy phân tích hình ảnh này và trả lời câu hỏi sau: {translated_message}",
                image_path,
                self.conversation_history,
            )
        else:
            gemini_response = send_to_gemini(
                translated_message, conversation_history=self.conversation_history
            )

        # Chỉ lưu phản hồi của bot nếu không phải là phản hồi cho lệnh
        if not message.startswith("/"):
            try:
                self.save_to_longterm_memory("assistant", gemini_response)
                self.conversation_history.append(
                    {"role": "assistant", "content": gemini_response}
                )
            except Exception as e:
                print(
                    error_color(
                        f"[!] Lỗi khi lưu phản hồi vào bộ nhớ dài hạn: {str(e)}"
                    )
                )

        emoji_response = emoji.emojize(gemini_response, language="alias")

        print(success_color(f"Phản hồi từ Gemini: {gemini_response}"))

        translated_response = self.translate_from_vietnamese(gemini_response)
        formatted_response = self.format_message_for_messenger(translated_response)

        if len(formatted_response) > self.MAX_MESSAGE_LENGTH:
            self.send_long_message(formatted_response)
        else:
            self.send_message(formatted_response)

    def handle_url_tools(self, message):
        """Xử lý các lệnh URL Tools"""
        try:
            if not hasattr(self, "url_tools"):
                self.url_tools = URLTools()

            parts = message.split(maxsplit=1)
            command = parts[0].lower()

            if len(parts) < 2 and command not in ["/urlhistory"]:
                return self.send_message(
                    """
                🔗 URL Tools - Hướng dẫn sử dụng:

                /short [url] - Rút gọn link
                /qr [nội dung] - Tạo mã QR
                /analyze [url] - Phân tích thông tin URL
                /urlhistory - Xem lịch sử thao tác

                Ví dụ:
                /short https://example.com/very/long/url
                /qr https://example.com
                /analyze https://example.com
                """
                )

            content = parts[1] if len(parts) > 1 else ""

            if command == "/short":
                result = self.url_tools.create_short_url(content)
                self.send_message(result)

            elif command == "/qr":
                qr_file = self.url_tools.create_qr_code(content)
                if not qr_file.startswith("❌"):
                    self.send_message("🔄 Đang tạo mã QR...")
                    self.send_image(qr_file)
                    os.remove(qr_file)  # Xóa file tạm
                else:
                    self.send_message(qr_file)

            elif command == "/analyze":
                result = self.url_tools.analyze_url(content)
                self.send_message(result)

            elif command == "/urlhistory":
                result = self.url_tools.get_history()
                self.send_message(result)

        except Exception as e:
            self.send_message(f"❌ Lỗi xử lý lệnh: {str(e)}")

    def change_language(self, message):
        parts = message.split()
        if len(parts) == 2:
            lang = parts[1].lower()
            if lang in self.supported_languages:
                self.current_language = lang
                # Xóa cache khi đổi ngôn ngữ
                if hasattr(self, "_translation_cache"):
                    self._translation_cache.clear()
                self.send_message(
                    self.translate_message(
                        f"Ngôn ngữ đã được thay đổi thành {self.supported_languages[lang]}."
                    )
                )
            else:
                supported_langs = "\n".join(
                    [
                        f"- {code}: {name}"
                        for code, name in self.supported_languages.items()
                    ]
                )
                self.send_message(
                    self.translate_message(
                        f"Ngôn ngữ không hợp lệ. Các ngôn ngữ được hỗ trợ:\n{supported_langs}"
                    )
                )
        else:
            self.send_message(
                self.translate_message(
                    "Lệnh không hợp lệ. Sử dụng /language [mã ngôn ngữ]."
                )
            )

    def _get_translation_cache(self):
        if not hasattr(self, "_translation_cache"):
            self._translation_cache = {}
        return self._translation_cache

    def _cache_translation(self, source_lang, target_lang, text, translation):
        cache = self._get_translation_cache()
        cache_key = f"{source_lang}:{target_lang}:{text}"
        cache[cache_key] = {"translation": translation, "timestamp": time.time()}

    def _get_cached_translation(self, source_lang, target_lang, text):
        cache = self._get_translation_cache()
        cache_key = f"{source_lang}:{target_lang}:{text}"

        if cache_key in cache:
            # Cache hết hạn sau 1 giờ
            if time.time() - cache[cache_key]["timestamp"] < 3600:
                return cache[cache_key]["translation"]
            else:
                del cache[cache_key]
        return None

    def translate_with_retry(self, text, source_lang, target_lang, max_retries=3):
        if not text or text.isspace():
            return text

        # Kiểm tra cache
        cached = self._get_cached_translation(source_lang, target_lang, text)
        if cached:
            return cached

        for attempt in range(max_retries):
            try:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                translation = translator.translate(text)

                if translation:
                    # Lưu vào cache
                    self._cache_translation(source_lang, target_lang, text, translation)
                    return translation

            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        error_color(
                            f"Lỗi dịch ({source_lang}->{target_lang}): {str(e)}"
                        )
                    )
                    return text  # Trả về text gốc nếu không dịch được
                time.sleep(1)  # Đợi 1 giây trước khi thử lại

        return text

    def translate_message(self, message):
        if self.current_language == "vi":
            return message
        return self.translate_with_retry(message, "vi", self.current_language)

    def translate_to_vietnamese(self, message):
        if self.current_language == "vi":
            return message
        return self.translate_with_retry(message, self.current_language, "vi")

    def translate_from_vietnamese(self, message):
        if self.current_language == "vi":
            return message
        return self.translate_with_retry(message, "vi", self.current_language)

    def format_message_for_messenger(self, message):

        code_blocks = re.findall(r"```[\s\S]*?```", message)
        for i, block in enumerate(code_blocks):
            placeholder = f"CODE_BLOCK_{i}"
            message = message.replace(block, placeholder)

        formatted = re.sub(r"\*\*(.*?)\*\*", r"*\1*", message)  # Bold
        formatted = re.sub(r"_(.*?)_", r"_\1_", formatted)  # Italic
        formatted = re.sub(r"`(.*?)`", r'"\1"', formatted)  # Inline code
        formatted = re.sub(r"\[(.*?)\]$$(.*?)$$", r"\1 (\2)", formatted)  # Links

        for i, block in enumerate(code_blocks):
            indented_block = "\n" + "\n".join(
                "    " + line for line in block.strip("`").split("\n")
            )
            formatted = formatted.replace(f"CODE_BLOCK_{i}", indented_block)

        lines = formatted.split("\n")
        for i, line in enumerate(lines):
            if re.match(r"^\d+\.\s", line):
                lines[i] = re.sub(r"^\d+\.\s", "• ", line)
            elif re.match(r"^[-*]\s", line):
                lines[i] = re.sub(r"^[-*]\s", "• ", line)

        formatted = "\n".join(lines)

        # Chuẩn hóa Unicode để tránh lỗi mã hóa
        formatted = unicodedata.normalize("NFKC", formatted)
        formatted = emoji.emojize(formatted, language="alias")

        return formatted

    def create_gist(self, content, description="Phản hồi dài từ Loki Bot"):
        print(info_color("[*] Đang tạo Gist..."))
        url = "https://api.github.com/gists"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        data = {
            "description": description,
            "public": False,
            "files": {"message.txt": {"content": content}},
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            gist_url = response.json()["html_url"]
            print(success_color(f"[*] Đã tạo Gist thành công: {gist_url}"))
            return gist_url
        except requests.exceptions.RequestException as e:
            print(error_color(f"[!] Lỗi khi tạo Gist: {str(e)}"))
            return None

    def send_audio(self, audio_path):
        """Gửi file audio qua messenger"""
        try:
            # Tìm nút đính kèm file
            attach_button = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]'))
            )

            # Gửi đường dẫn file audio
            attach_button.send_keys(os.path.abspath(audio_path))

            # Đợi và click nút gửi
            send_button = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'div[aria-label="Gửi"]')
                )
            )
            send_button.click()

            print(success_color(f"[*] Đã gửi file audio: {audio_path}"))

        except Exception as e:
            print(error_color(f"[!] Lỗi gửi file audio: {str(e)}"))
            self.send_message("❌ Không thể gửi file audio")

    def send_long_message(self, message):
        self.message_counter += 1
        print(info_color(f"[*] Xử lý tin nhắn dài #{self.message_counter}"))

        gist_url = self.create_gist(message)
        if gist_url:
            short_message = self.translate_message(
                f"Phản hồi quá dài. Xem nội dung đầy đủ tại đây: {gist_url}"
            )
            self.send_message(short_message)
        else:
            print(
                warning_color("[!] Không thể tạo Gist. Chuyển sang gửi nhiều tin nhắn.")
            )
            chunks = [
                message[i : i + self.MAX_MESSAGE_LENGTH]
                for i in range(0, len(message), self.MAX_MESSAGE_LENGTH)
            ]
            for i, chunk in enumerate(chunks, 1):
                chunk_message = self.translate_message(
                    f"Phần {i}/{len(chunks)}:\n\n{chunk}"
                )
                self.send_message(chunk_message)
                time.sleep(1)  # Tránh gửi quá nhanh

    def new_image_listen(self, return_file=False, file_path=None, return_byte=True):
        if self.current_img_value != self.his_img_value:
            self.his_img_value = self.current_img_value
            if return_file:
                if file_path is None:
                    self.image_counter += 1
                    file_path = f"./meschat_img_{self.image_counter}.png"
                img = io.BytesIO(self.current_image)
                img = Image.open(img)
                img.save(file_path)
                return file_path
            elif return_byte:
                return self.current_image
        else:
            return None

    def save_conversation_history(self, filename="conversation_history.json"):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(success_color(f"[*] Đã lưu lịch sử trò chuyện vào {filename}"))
        except Exception as e:
            print(error_color(f"[!] Lỗi khi lưu lịch sử trò chuyện: {str(e)}"))

    def load_conversation_history(self, filename="conversation_history.json"):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                self.conversation_history = json.load(f)
            print(success_color(f"Đã tải lịch sử trò chuyện từ {filename}"))
        except FileNotFoundError:
            print(error_color(f"Không tìm thấy file {filename}"))

    def save_to_longterm_memory(self, role, content):
        """Lưu tin nhắn vào bộ nhớ dài hạn"""
        try:
            # Tạo bản ghi mới
            memory_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "role": role,
                "content": content,
            }

            # Đọc file hiện tại hoặc tạo list mới nếu file không tồn tại
            try:
                with open(self.longterm_memory_file, "r", encoding="utf-8") as f:
                    memories = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                memories = []

            # Thêm bản ghi mới
            memories.append(memory_entry)

            # Giới hạn kích thước bộ nhớ
            if len(memories) > self.max_memory_size:
                memories = memories[-self.max_memory_size :]

            # Lưu vào file
            with open(self.longterm_memory_file, "w", encoding="utf-8") as f:
                json.dump(memories, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(error_color(f"[!] Lỗi khi lưu vào bộ nhớ dài hạn: {str(e)}"))

    def summarize_conversations(self, time_range="today"):
        """Tóm tắt cuộc trò chuyện theo khoảng thời gian"""
        try:
            if not os.path.exists(self.longterm_memory_file):
                return self.translate_message("Chưa có dữ liệu để tóm tắt.")

            with open(self.longterm_memory_file, "r", encoding="utf-8") as f:
                memories = json.load(f)

            # Lọc tin nhắn theo thời gian
            now = datetime.now()
            if time_range == "today":
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif time_range == "week":
                start_time = now - timedelta(days=7)
            elif time_range == "month":
                start_time = now - timedelta(days=30)
            else:
                start_time = datetime.min

            filtered_messages = []
            for memory in memories:
                try:
                    msg_time = datetime.strptime(
                        memory["timestamp"], "%Y-%m-%d %H:%M:%S"
                    )
                    if msg_time >= start_time:
                        filtered_messages.append(
                            f"{memory['role']}: {memory['content']}"
                        )
                except (ValueError, KeyError):
                    continue

            if not filtered_messages:
                return self.translate_message(
                    f"Không có tin nhắn nào trong {time_range}"
                )

            # Tạo prompt cho Gemini
            prompt = f"""Hãy tóm tắt cuộc tr�� chuyện sau một cách ngắn gọn và súc tích:

Thời gian: {time_range}
Số lượng tin nhắn: {len(filtered_messages)}

Nội dung trò chuyện:
{chr(10).join(filtered_messages[-50:])}  # Chỉ lấy 50 tin nhắn gần nhất để tránh quá tải

Yêu cầu tóm tắt:
1. Các chủ đề chính được thảo luận
2. Những điểm quan trọng/quyết định
3. Tâm trạng/cảm xúc chung của cuộc trò chuyện
4. Các yêu cầu hoặc vấn đề cần follow-up

Hãy trình bày tóm tắt một cách súc tích và dễ đọc."""

            # Gọi Gemini để tóm tắt
            summary = send_to_gemini(prompt)

            # Lưu tóm tắt
            self.memory_summary[time_range] = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": summary,
            }

            return summary

        except Exception as e:
            error_msg = f"Lỗi khi tóm tắt cuộc trò chuyện: {str(e)}"
            print(error_color(f"[!] {error_msg}"))
            return self.translate_message(f"⚠️ {error_msg}")

    def send_random_dog_image(self):
        def wait_for_upload(seconds=3):
            """Đợi để đảm bảo ảnh đưc tải lên hoàn toàn"""
            time.sleep(seconds)

        def attempt_send_image(image_path, max_retries=3, base_delay=5):
            """Thử gửi ảnh với cơ chế retry"""
            for attempt in range(max_retries):
                try:
                    self.send_image(image_path)
                    wait_for_upload()  # Đợi ảnh tải lên hoàn tất
                    return True
                except Exception as e:
                    delay = base_delay * (
                        attempt + 1
                    )  # Tăng thời gian chờ sau mỗi lần thất bại
                    print(
                        warning_color(
                            f"[!] Lần thử {attempt + 1}/{max_retries} thất bại: {str(e)}"
                        )
                    )
                    print(info_color(f"[*] Đợi {delay} giây trước khi thử lại..."))
                    time.sleep(delay)
            return False

        try:
            dog_images_dir = "/content/drive/MyDrive/"
            if not os.path.exists(dog_images_dir):
                self.send_message(
                    self.translate_message("❌ Không tìm thấy thư mục chứa ảnh.")
                )
                return

            # Lọc và kiểm tra ảnh hợp lệ
            dog_images = [
                f
                for f in os.listdir(dog_images_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
                and os.path.getsize(os.path.join(dog_images_dir, f)) < 10000000
            ]  # Giới hạn 10MB

            if not dog_images:
                self.send_message(
                    self.translate_message(
                        "❌ Không tìm thấy ảnh phù hợp trong thư viện."
                    )
                )
                return

            # Chọn ảnh ngẫu nhiên
            num_images = random.randint(5, 10)  # Giảm số lượng ảnh để đảm bảo ổn định
            selected_images = random.sample(
                dog_images, min(num_images, len(dog_images))
            )
            total_images = len(selected_images)

            # Thông báo bắt đầu
            self.send_message(
                self.translate_message(
                    f"🔄 Đang chuẩn bị tải lên {total_images} hình ảnh...\n"
                    f"⏳ Vui lòng đợi trong giây lát..."
                )
            )
            wait_for_upload(5)  # Đợi thông báo hiển thị

            # Theo dõi tiến trình
            uploaded_count = 0
            failed_uploads = 0

            # Tải lên từng ảnh
            for index, image in enumerate(selected_images, 1):
                image_path = os.path.join(dog_images_dir, image)

                # Kiểm tra file tồn tại và có thể đọc được
                if not os.path.exists(image_path) or not os.access(image_path, os.R_OK):
                    print(error_color(f"[!] Không thể truy cập file: {image}"))
                    failed_uploads += 1
                    continue

                print(info_color(f"[*] Đang xử lý ảnh {index}/{total_images}: {image}"))

                if attempt_send_image(image_path):
                    uploaded_count += 1
                    print(success_color(f"[*] Tải lên thành công: {image}"))
                else:
                    failed_uploads += 1
                    print(
                        error_color(f"[!] Không thể tải lên sau nhiều lần thử: {image}")
                    )

                # Đợi giữa các lần tải để tránh quá tải
                wait_for_upload(7)  # Tăng thời gian chờ giữa các ảnh

            # Gửi thông báo kết quả
            if failed_uploads == 0:
                final_message = self.translate_message(
                    f"✅ Hoàn thành!\n"
                    f"📤 Đã tải lên thành công {uploaded_count}/{total_images} ảnh."
                )
            else:
                final_message = self.translate_message(
                    f"⚠️ Đã hoàn thành với một số lỗi:\n"
                    f"✅ Thành công: {uploaded_count} ảnh\n"
                    f"❌ Thất bại: {failed_uploads} ảnh"
                )

            wait_for_upload(5)  # Đợi tất cả ảnh hiển thị
            self.send_message(final_message)

        except Exception as e:
            error_message = f"❌ Đã xảy ra lỗi không mong muốn: {str(e)}"
            print(error_color(f"[!] {error_message}"))
            self.send_message(self.translate_message(error_message))

    def get_useless_fact(self):  # fact đủ thứ
        try:
            response = requests.get(
                "https://uselessfacts.jsph.pl/random.json?language=en"
            )
            response.raise_for_status()
            return response.json()["text"]
        except requests.RequestException:
            return None

    def get_number_fact(self):
        try:
            response = requests.get("http://numbersapi.com/random/trivia")
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    def get_cat_fact(self):
        try:
            response = requests.get("https://catfact.ninja/fact")
            response.raise_for_status()
            return response.json()["fact"]
        except requests.RequestException:
            return None

    def get_dog_fact(self):
        try:
            response = requests.get("https://dog-api.kinduff.com/api/facts")
            response.raise_for_status()
            return response.json()["facts"][0]
        except requests.RequestException:
            return None

    def get_today_in_history(self):
        try:
            today = datetime.now()
            response = requests.get(
                f"https://history.muffinlabs.com/date/{today.month}/{today.day}"
            )
            response.raise_for_status()
            events = response.json()["data"]["Events"]
            event = random.choice(events)
            return f"On this day in {event['year']}: {event['text']}"
        except (requests.RequestException, KeyError, IndexError):
            return None

    def get_random_fact(self):
        while True:
            fact_function = random.choice(self.fact_apis)
            fact = fact_function()
            if fact and fact not in self.used_facts:
                self.used_facts.append(fact)
                return fact

    def translate_and_format_fact(self, fact):
        # Translate to Vietnamese
        fact_in_vietnamese = GoogleTranslator(source="en", target="vi").translate(fact)

        # Translate to the current language if it's not Vietnamese
        if self.current_language != "vi":
            translated_fact = GoogleTranslator(
                source="vi", target=self.current_language
            ).translate(fact_in_vietnamese)
        else:
            translated_fact = fact_in_vietnamese

        return f"Fact thú vị: {translated_fact}"

    def send_random_fact(self):
        if self.fact_mode:
            fact = self.get_random_fact()
            if fact:
                formatted_fact = self.translate_and_format_fact(fact)
                self.send_message(formatted_fact)
            else:
                self.send_message(
                    self.translate_message(
                        "Xin lỗi, không thể lấy fact lúc này. Vui lòng thử lại sau."
                    )
                )

    def start_fact_loop(self):
        if self.fact_mode:
            self.send_random_fact()
            threading.Timer(self.fact_interval, self.start_fact_loop).start()

    def clear_bot_data(self):
        try:
            # Xóa lịch sử trò chuyện
            self.conversation_history = []

            # Đặt lại các biến đếm
            self.image_counter = 0
            self.message_counter = 0

            # Xóa các file ảnh tạm thời và dọn dẹp thư mục ảnh
            self.cleanup_images()

            # Đặt lại ngôn ngữ về mặc định
            previous_language = self.current_language
            self.current_language = "vi"

            # Xóa cache dịch thuật
            if hasattr(self, "_translation_cache"):
                self._translation_cache.clear()

            # Đặt lại thời gian bắt đầu
            self.start_time = datetime.now()

            # Xóa file lịch sử trò chuyện
            if os.path.exists("conversation_history.json"):
                os.remove("conversation_history.json")

            # Gửi thông báo kết quả
            if previous_language != "vi":
                self.send_message(
                    "✨ Đã dọn dẹp xong!\n"
                    + f"🔄 Chuyển ngôn ngữ từ {self.supported_languages.get(previous_language, previous_language)} "
                    + "về Tiếng Việt\n"
                    + "🤖 Bot đã sẵn sàng phục vụ!"
                )
            else:
                self.send_message("✨ Đã dọn dẹp xong!\n🤖 Bot đã sẵn sàng phục vụ!")

        except Exception as e:
            error_msg = f"❌ Lỗi khi dọn dẹp: {str(e)}"
            print(error_color(f"[!] {error_msg}"))
            self.send_message(self.translate_message(error_msg))

    def cleanup_images(self):
        """Dọn dẹp tất cả các thư mục ảnh và file ảnh tạm thời"""
        try:
            # Dọn dẹp thư mục ảnh tải về
            image_dir = "downloaded_images"
            if os.path.exists(image_dir):
                # Giữ lại 50 ảnh gần nhất
                images = [
                    f
                    for f in os.listdir(image_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
                ]
                images.sort(
                    key=lambda x: os.path.getctime(os.path.join(image_dir, x)),
                    reverse=True,
                )

                # Xóa các ảnh cũ
                for old_image in images[50:]:
                    try:
                        os.remove(os.path.join(image_dir, old_image))
                        print(info_color(f"[*] Đã xóa ảnh cũ: {old_image}"))
                    except Exception as e:
                        print(warning_color(f"[!] Không thể xóa {old_image}: {str(e)}"))

            # Xóa các file ảnh tạm thời
            temp_patterns = [
                "meschat_img_*.png",
                "optimized_*.jpg",
                "generated_image.jpg",
                "temp_image.jpg",
            ]

            for pattern in temp_patterns:
                for file in glob.glob(pattern):
                    try:
                        os.remove(file)
                        print(info_color(f"[*] Đã xóa file tạm: {file}"))
                    except Exception as e:
                        print(warning_color(f"[!] Không thể xóa {file}: {str(e)}"))

            # Đặt lại đường dẫn ảnh cuối
            self.last_image_path = None

            print(success_color("[*] Hoàn thành dọn dẹp ảnh"))

        except Exception as e:
            print(error_color(f"[!] Lỗi khi dọn dẹp ảnh: {str(e)}"))

    def clear_temporary_images(self):
        # Xóa tất cả các file ảnh tạm thời
        for file in os.listdir():
            if file.startswith("meschat_img_") and file.endswith(".png"):
                os.remove(file)

        self.last_image_path = None

    def handle_image(self, img_path):
        try:
            # Tạo tên file tạm thời duy nhất
            temp_filename = f"optimized_{int(time.time())}_{os.path.basename(img_path)}"

            with Image.open(img_path) as img:
                # Giới hạn kích thước tối đa
                max_size = (1280, 1280)
                img.thumbnail(max_size, Image.LANCZOS)

                # Tối ưu chất lượng và lưu với tên file tạm thời
                img.save(temp_filename, quality=85, optimize=True)

            return temp_filename

        except Exception as e:
            print(error_color(f"[!] Lỗi khi xử lý ảnh: {str(e)}"))
            return img_path  # Trả về đường dẫn gốc nếu có lỗi

    def send_guide(self):
        """Gửi link hướng dẫn sử dụng bot"""
        guide_url = "YOUR_GUIDE_URL"
        message = self.translate_message(f"📚 Xem hướng dẫn chi tiết tại:\n{guide_url}")
        self.send_message(message)


CHAT_URLS = {
    "test": "https://www.messenger.com/t/YOUR_TEST_CHAT_ID",
    "group1": "https://www.messenger.com/t/YOUR_GROUP1_ID",
    "group2": "https://www.messenger.com/t/YOUR_GROUP2_ID",
}

if __name__ == "__main__":
    CURRENT_CHAT = "test"  # Change this to your desired chat
    while True:
        try:
            # Khởi tạo bot
            meschat = MesChat(
                email_or_phone="YOUR_EMAIL",
                password="YOUR_PASSWORD",
                group_or_chat=CHAT_URLS[CURRENT_CHAT],
            )

            print(
                highlight_color(
                    "Bot đã sẵn sàng. Đang lắng nghe tin nhắn và hình ảnh..."
                )
            )

            # Vòng lặp chính để lắng nghe tin nhắn
            while True:
                try:
                    new_message = meschat.new_message_listen()
                    if new_message is not None:
                        meschat.handle_new_message(new_message)
                    time.sleep(1)

                except Exception as e:
                    print(error_color(f"deo nhan duoc tin nhan roi: {str(e)}"))
                    # log nếu cần
                    print(error_color(f"duma lỗi: {type(e).__name__}"))
                    time.sleep(5)  # cc

        except Exception as e:
            print(error_color(f"Lỗi khởi tạo bot: {str(e)}"))
            print(error_color(f"Đang thử kết nối lại sau 60 giây..."))
            time.sleep(60)
