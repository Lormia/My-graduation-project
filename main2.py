import os
import torch
import clip
import requests
import re
import time
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from threading import Thread
from math import ceil


class ImageSearchApp:
    def __init__(self, root, switch_callback=None):
        self.root = root
        self.switch_callback = switch_callback  # 切换回调函数
        self.root.title("图像检索系统 - 百度图片搜索")
        self.root.geometry("1000x800")

        # 设置样式
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Treeview', font=('Arial', 10), rowheight=25)

        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 搜索部分
        self.search_frame = ttk.LabelFrame(self.main_frame, text="搜索设置", padding=10)
        self.search_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.search_frame, text="关键词:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.keyword_entry = ttk.Entry(self.search_frame, width=30)
        self.keyword_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))

        ttk.Label(self.search_frame, text="图片数量:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.count_entry = ttk.Entry(self.search_frame, width=10)
        self.count_entry.grid(row=0, column=3, sticky=tk.W)
        self.count_entry.insert(0, "20")  # 默认值

        self.search_btn = ttk.Button(self.search_frame, text="开始搜索", command=self.start_search)
        self.search_btn.grid(row=0, column=4, padx=(10, 0))

        # 添加切换按钮
        if switch_callback:
            self.switch_btn = ttk.Button(
                self.search_frame,
                text="切换到本地检索",
                command=self.switch_to_local,
                style='Switch.TButton'
            )
            self.switch_btn.grid(row=0, column=5, padx=(10, 0))
            self.style.configure('Switch.TButton', foreground='blue')

        # 进度显示
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_label = ttk.Label(self.progress_frame, text="准备就绪")
        self.progress_label.pack(side=tk.LEFT)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, expand=True)

        # 结果展示区域
        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        # 图片列表区域
        self.list_frame = ttk.LabelFrame(self.result_frame, text="匹配图片列表", width=300)
        self.list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 创建Treeview显示图片列表
        self.tree = ttk.Treeview(self.list_frame, columns=('相似度'), selectmode='browse')
        self.tree.heading('#0', text='图片名称')
        self.tree.heading('相似度', text='相似度')
        self.tree.column('#0', width=200)
        self.tree.column('相似度', width=80, anchor='center')

        vsb = ttk.Scrollbar(self.list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # 绑定选择事件
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)

        # 图片显示区域
        self.image_frame = ttk.LabelFrame(self.result_frame, text="图片预览")
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 底部信息栏
        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.pack(fill=tk.X, pady=(5, 0))

        self.time_label = ttk.Label(self.info_frame, text="耗时: -")
        self.time_label.pack(side=tk.LEFT)

        self.similarity_label = ttk.Label(self.info_frame, text="相似度: -")
        self.similarity_label.pack(side=tk.LEFT, padx=(20, 0))

        self.selected_label = ttk.Label(self.info_frame, text="选中: -")
        self.selected_label.pack(side=tk.LEFT, padx=(20, 0))

        # 初始化CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # 存储结果
        self.similar_dict = {}
        self.search_thread = None
        self.image_items = []

    def switch_to_local(self):
        """切换到本地图片检索模式"""
        if self.switch_callback:
            self.root.destroy()
            self.switch_callback()

    def start_search(self):
        keyword = self.keyword_entry.get().strip()
        count = self.count_entry.get().strip()

        if not keyword:
            messagebox.showerror("错误", "请输入搜索关键词")
            return

        try:
            count = int(count)
            if count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "请输入有效的图片数量(正整数)")
            return

        # 清除旧结果
        self.tree.delete(*self.tree.get_children())
        self.image_label.config(image='')
        self.image_label.image = None
        self.similar_dict = {}
        self.image_items = []

        # 禁用按钮防止重复点击
        self.search_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="正在搜索...")
        self.progress_var.set(0)

        # 在新线程中执行搜索
        self.search_thread = Thread(target=self.perform_search, args=(keyword, count))
        self.search_thread.start()

        # 检查线程是否完成
        self.root.after(100, self.check_thread)

    def check_thread(self):
        if self.search_thread.is_alive():
            self.root.after(100, self.check_thread)
        else:
            self.search_btn.config(state=tk.NORMAL)

    def perform_search(self, keyword, count):
        time_start = time.time()

        # 创建图片保存目录
        image_folder = './image'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # 记录下载前已存在的图片，以便后续排除
        existing_images_before = set(f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png')))

        headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive',
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
            'Host': 'image.baidu.com',
            'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=&st=-1&fm=result&fr=&sf=1&fmq=1610952036123_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=%E6%98%9F%E9%99%85',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'X-Requested-With': 'XMLHttpRequest'
        }

        url = 'http://image.baidu.com/search/index?tn=baiduimage&fm=result&ie=utf-8&word='
        url = url + keyword + "&pn="

        self.update_progress("正在获取图片列表...", 10)

        try:
            strhtml = requests.get(url, headers=headers)
            string = str(strhtml.text)

            img_url_regex = '"thumbURL":"(.*?)",'
            count_downloaded = 0
            index = 0
            downloaded_images = []  # 存储本次下载的图片路径

            while count_downloaded < count:
                strhtml = requests.get(url + str(index), headers=headers)
                string = str(strhtml.text)
                pic_url = re.findall(img_url_regex, string)
                index += len(pic_url)

                if not pic_url:
                    break

                for i, each in enumerate(pic_url):
                    if count_downloaded >= count:
                        break

                    progress = 10 + int(90 * (count_downloaded / count))
                    self.update_progress(f"正在下载第 {count_downloaded + 1}/{count} 张图片", progress)

                    try:
                        if each is not None:
                            pic = requests.get(each, timeout=5)
                            # 使用时间戳确保文件名唯一
                            timestamp = int(time.time() * 1000)
                            image_name = f'{keyword}_{timestamp}_{count_downloaded + 1}.jpg'
                            image_path = os.path.join(image_folder, image_name)
                            with open(image_path, 'wb') as fp:
                                fp.write(pic.content)
                            downloaded_images.append(image_path)
                            count_downloaded += 1
                    except Exception as e:
                        print(f'下载错误: {e}')
                        continue

            self.update_progress("正在分析图片相似度...", 95)

            # 只分析本次下载的图片
            self.similar_dict = {}
            for image_path in downloaded_images:
                sim = self.match(image_path, keyword)
                self.similar_dict[image_path] = sim

            # 按相似度排序
            sorted_items = sorted(self.similar_dict.items(), key=lambda x: x[1], reverse=True)

            # 更新Treeview
            self.tree.delete(*self.tree.get_children())  # 清空现有列表
            self.image_items = []  # 清空现有图片项

            for i, (path, sim) in enumerate(sorted_items):
                filename = os.path.basename(path)
                sim_value = f"{float(sim):.4f}"
                item_id = self.tree.insert('', 'end', text=filename, values=(sim_value))
                self.image_items.append((item_id, path))

                # 默认选择第一个
                if i == 0:
                    self.tree.selection_set(item_id)
                    self.show_image(path)

            time_end = time.time()
            time_used = f"{time_end - time_start:.2f}秒"

            self.update_progress(f"搜索完成，共找到 {len(sorted_items)} 张图片", 100)
            self.update_info(time_used, "-", "-")

        except Exception as e:
            self.update_progress(f"错误: {str(e)}", 100)
            messagebox.showerror("错误", f"搜索过程中发生错误: {str(e)}")

    def on_tree_select(self, event):
        selected_item = self.tree.selection()
        if selected_item:
            item_id = selected_item[0]
            # 查找对应的图片路径
            for img_item in self.image_items:
                if img_item[0] == item_id:
                    path = img_item[1]
                    sim_value = self.tree.item(item_id, 'values')[0]
                    self.show_image(path)
                    self.update_info(self.time_label.cget('text'), sim_value, os.path.basename(path))
                    break

    def update_progress(self, message, value):
        self.root.after(0, lambda: self._update_progress(message, value))

    def _update_progress(self, message, value):
        self.progress_label.config(text=message)
        self.progress_var.set(value)

    def update_info(self, time_used, similarity, selected=None):
        self.time_label.config(text=f"{time_used}")
        self.similarity_label.config(text=f"相似度: {similarity}")
        if selected:
            self.selected_label.config(text=f"选中: {selected}")

    def match(self, imagePath, sentence):
        image = self.preprocess(Image.open(imagePath)).unsqueeze(0).to(self.device)
        text = clip.tokenize([sentence]).to(self.device)
        with torch.no_grad():
            logits_per_image, _ = self.model(image, text)
            similarity = float(logits_per_image[0][0])
            return similarity

    def show_image(self, image_path):
        try:
            img = Image.open(image_path)
            # 调整图片大小以适应窗口
            max_width = self.image_frame.winfo_width() - 40
            max_height = self.image_frame.winfo_height() - 40

            if img.width > max_width or img.height > max_height:
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持引用

        except Exception as e:
            messagebox.showerror("错误", f"无法显示图片: {str(e)}")


if __name__ == '__main__':
    root = tk.Tk()


    # 在没有切换回调的情况下运行
    def no_switch():
        messagebox.showinfo("提示", "这是独立运行模式，切换功能不可用")


    app = ImageSearchApp(root, switch_callback=no_switch)
    root.mainloop()