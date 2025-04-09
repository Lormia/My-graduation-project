import tkinter as tk
from main2 import ImageSearchApp
from main import CLIPImageSearchApp


class AppSwitcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏主窗口
        self.open_baidu_search()  # 默认打开百度搜索

    def open_baidu_search(self):
        """打开在线搜索界面"""
        window = tk.Toplevel()
        window.protocol("WM_DELETE_WINDOW", self.on_close)
        ImageSearchApp(window, switch_callback=self.open_clip_search)

    def open_clip_search(self):
        """打开本地图片检索界面"""
        window = tk.Toplevel()
        window.protocol("WM_DELETE_WINDOW", self.on_close)
        CLIPImageSearchApp(window, switch_callback=self.open_baidu_search)

    def on_close(self):
        """关闭所有窗口"""
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    AppSwitcher()
    tk.mainloop()