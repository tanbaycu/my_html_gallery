from colorama import Fore, Style
import time
import random

def cay_thong_noel():
    # Kích thước cây
    chieu_cao = 20
    
    # Các màu cho đèn trang trí
    mau = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]
    
    while True:
        print("\033[2J\033[H") # Xóa màn hình
        
        # Vẽ ngôi sao
        print(" " * (chieu_cao-1) + Fore.YELLOW + "★" + Style.RESET_ALL)
        
        # Vẽ thân cây
        for i in range(chieu_cao):
            khoang_trang = " " * (chieu_cao - i - 1)
            la = ""
            for j in range(2*i + 1):
                if random.random() < 0.1:  # 10% cơ hội xuất hiện đèn
                    la += random.choice(mau) + "o" + Style.RESET_ALL
                else:
                    la += Fore.GREEN + "*" + Style.RESET_ALL
            print(khoang_trang + la)
        
        # Vẽ thân gỗ
        for i in range(2):
            print(" " * (chieu_cao-2) + Fore.RED + "| |" + Style.RESET_ALL)
            
        # Chờ 0.5 giây trước khi cập nhật
        time.sleep(0.5)

if __name__ == "__main__":
    cay_thong_noel()
