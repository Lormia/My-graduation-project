import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import torch
import clip
from sklearn.neighbors import NearestNeighbors
import numpy as np
import threading
import pickle
import time
import re
import jieba
from torch.cuda.amp import autocast
from torchvision import transforms


class CLIPImageSearchApp:
    def __init__(self, root, switch_callback=None):
        self.root = root
        self.switch_callback = switch_callback  # 切换回调函数
        self.root.title("图像检索系统 - 本地图片检索")
        self.root.geometry("1000x700")

        # 模型配置
        self.available_models = {
            "ViT-B/32": "平衡精度和速度",
            "ViT-L/14": "高精度但较慢"
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = None, None
        self.current_model_name = None

        # 数据存储
        self.image_features = None
        self.image_paths = []
        self.knn_model = None
        self.feature_file = "image_features.pkl"

        # 文本处理工具初始化
        jieba.initialize()
        self.stopwords = set(["的", "了", "在", "和", "是", "我"])

        # 创建GUI
        self.create_widgets()

        # 默认加载ViT-B/32模型
        self.load_model("ViT-L/14")
        self.current_result_indices = None

    def preprocess_text(self, text):
        """增强文本预处理"""
        # 移除特殊字符和标点
        text = re.sub(r'[^\w\u4e00-\u9fff\s]', '', text)

        # 中文分词
        words = jieba.lcut(text)

        # 扩展停用词列表
        extended_stopwords = self.stopwords.union({
            "这个", "那个", "一些", "一种", "一张", "一幅"
        })

        # 过滤停用词并保留有意义词汇
        meaningful_words = [w for w in words if w not in extended_stopwords and len(w) > 1]

        # 添加CLIP提示词增强
        if meaningful_words:
            return "一张清晰的" + " ".join(meaningful_words) + "的照片"
        return text

    def load_model(self, model_name="ViT-L/14"):
        """加载模型并增强预处理"""
        try:
            self.status_var.set(f"正在加载模型 {model_name}...")
            self.root.update_idletasks()

            start_time = time.time()
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.current_model_name = model_name

            # 增强图像预处理
            self.preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
            ])

            load_time = time.time() - start_time
            self.status_var.set(f"{model_name} 加载完成 (耗时 {load_time:.1f}秒)")

        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")

    def create_widgets(self):
        """创建GUI界面"""
        # 顶部框架 - 控制区域
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)

        # 添加切换按钮（放在最左侧）
        if self.switch_callback:
            switch_btn = tk.Button(
                control_frame,
                text="切换到网络搜索",
                command=self.switch_to_online,
                fg='blue'
            )
            switch_btn.pack(side=tk.LEFT, padx=5)

        # 模型选择按钮
        model_button = tk.Button(control_frame, text="选择模型", command=self.select_model)
        model_button.pack(side=tk.LEFT, padx=5)

        # 搜索区域
        search_frame = tk.Frame(control_frame)
        search_frame.pack(side=tk.LEFT, padx=20)

        tk.Label(search_frame, text="搜索文本:").pack(side=tk.LEFT)
        self.search_entry = tk.Entry(search_frame, width=50)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind("<Return>", lambda event: self.search_images())

        search_button = tk.Button(search_frame, text="搜索", command=self.search_images)
        search_button.pack(side=tk.LEFT, padx=5)

        # 图片库操作按钮
        lib_frame = tk.Frame(control_frame)
        lib_frame.pack(side=tk.LEFT)

        load_button = tk.Button(lib_frame, text="加载图片库", command=self.load_image_library)
        load_button.pack(side=tk.LEFT, padx=5)

        save_button = tk.Button(lib_frame, text="保存特征", command=self.save_features)
        save_button.pack(side=tk.LEFT, padx=5)

        load_feat_button = tk.Button(lib_frame, text="加载特征", command=self.load_features)
        load_feat_button.pack(side=tk.LEFT, padx=5)

        # 进度条
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=10)
        self.progress.pack_forget()

        # 结果展示区域
        result_frame = tk.Frame(self.root)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧 - 搜索结果列表
        list_frame = tk.Frame(result_frame, width=300)
        list_frame.pack(side=tk.LEFT, fill=tk.Y)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_listbox = tk.Listbox(list_frame, width=40, height=20, yscrollcommand=scrollbar.set)
        self.result_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_listbox.bind("<<ListboxSelect>>", self.show_selected_image)

        scrollbar.config(command=self.result_listbox.yview)

        # 右侧 - 图片显示区域
        self.image_frame = tk.Frame(result_frame, width=700, height=600, bg='white')
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 底部状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪 - 请先选择模型并加载图片库")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def switch_to_online(self):
        """切换到百度图片搜索模式"""
        if self.switch_callback:
            self.root.destroy()
            self.switch_callback()

    def select_model(self):
        """选择CLIP模型"""
        model_name = simpledialog.askstring(
            "选择模型",
            "请选择CLIP模型:\n" + "\n".join([f"{k} - {v}" for k, v in self.available_models.items()]),
            initialvalue="ViT-B/32"
        )

        if model_name and model_name in self.available_models:
            self.load_model(model_name)

    def load_image_library(self):
        """选择并加载图片库"""
        if not self.model:
            messagebox.showwarning("警告", "请先选择模型")
            return

        folder_path = filedialog.askdirectory(title="选择图片库文件夹")
        if not folder_path:
            return

        self.status_var.set("正在加载图片库...")
        self.progress.pack(side=tk.LEFT, padx=10)
        self.progress["value"] = 0

        # 在新线程中处理
        threading.Thread(target=self.process_image_library, args=(folder_path,), daemon=True).start()

    def process_image_library(self, folder_path):
        """增强图片处理流程"""
        try:
            self.image_paths = []
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            error_log = []

            # 第一阶段：收集并验证图片
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        img_path = os.path.join(root, file)
                        try:
                            # 预验证图片可读性
                            with Image.open(img_path) as test_img:
                                test_img.verify()  # 验证文件完整性
                                if test_img.mode not in ('RGB', 'L'):
                                    test_img = test_img.convert('RGB')
                            self.image_paths.append(img_path)
                        except Exception as e:
                            error_log.append(f"验证失败: {img_path} - {str(e)}")
                            continue

            if not self.image_paths:
                self.status_var.set("错误: 没有有效图片文件")
                if error_log:
                    self.show_error_report(error_log)
                return

            # 第二阶段：特征提取
            self.image_features = []
            success_count = 0
            total_images = len(self.image_paths)

            for i, image_path in enumerate(self.image_paths):
                try:
                    # 进度更新
                    progress = (i + 1) / total_images * 100
                    self.progress["value"] = progress
                    self.status_var.set(f"处理中 {i + 1}/{total_images} - {os.path.basename(image_path)}")
                    self.root.update_idletasks()

                    # 增强的图像加载
                    with Image.open(image_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # 自动调整超大图像
                        if max(img.size) > 4096:
                            img.thumbnail((2048, 2048))

                        # 特征提取
                        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
                        with autocast(), torch.no_grad():
                            features = self.model.encode_image(image_input)

                        self.image_features.append(features.cpu().numpy())
                        success_count += 1

                except Exception as e:
                    error_log.append(f"处理失败: {image_path} - {str(e)}")
                    continue

            # 结果处理
            if success_count == 0:
                self.status_var.set("错误: 所有图片处理失败")
                self.show_error_report(error_log)
                return

            self.image_features = np.vstack(self.image_features)
            self.image_features /= np.linalg.norm(self.image_features, axis=1, keepdims=True)

            # 构建索引
            try:
                import faiss
                index = faiss.IndexFlatIP(self.image_features.shape[1])
                index.add(self.image_features)
                self.knn_model = index
                self.status_var.set(f"成功处理 {success_count}/{total_images} 图片 (FAISS加速)")
            except:
                self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
                self.knn_model.fit(self.image_features)
                self.status_var.set(f"成功处理 {success_count}/{total_images} 图片")

            # 显示错误报告
            if error_log:
                self.show_error_report(error_log[:20])  # 最多显示20条错误

        except Exception as e:
            self.status_var.set(f"系统错误: {str(e)}")
            self.progress.pack_forget()


    def save_features(self):
        """保存提取的特征到文件"""
        if self.image_features is None or len(self.image_paths) == 0:
            messagebox.showwarning("警告", "没有可保存的特征数据")
            return

        try:
            # 准备保存的数据
            save_data = {
                'model_name': self.current_model_name,
                'image_paths': self.image_paths,
                'image_features': self.image_features,
                'timestamp': time.time()
            }

            # 选择保存位置
            file_path = filedialog.asksaveasfilename(
                title="保存特征文件",
                initialfile=self.feature_file,
                defaultextension=".pkl",
                filetypes=[("Pickle文件", "*.pkl")]
            )

            if not file_path:
                return

            # 保存数据
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f)

            self.status_var.set(f"特征已保存到 {file_path}")

        except Exception as e:
            messagebox.showerror("错误", f"保存特征失败: {str(e)}")

    def load_features(self):
        """从文件加载特征"""
        file_path = filedialog.askopenfilename(
            title="选择特征文件",
            initialfile=self.feature_file,
            filetypes=[("Pickle文件", "*.pkl")]
        )

        if not file_path:
            return

        try:
            self.status_var.set("正在加载特征文件...")
            self.root.update_idletasks()

            with open(file_path, 'rb') as f:
                save_data = pickle.load(f)

            # 检查模型是否匹配
            if 'model_name' in save_data:
                if save_data['model_name'] != self.current_model_name:
                    answer = messagebox.askyesno(
                        "模型不匹配",
                        f"特征文件使用的是 {save_data['model_name']} 模型，当前模型是 {self.current_model_name}。\n是否切换到 {save_data['model_name']} 模型？"
                    )
                    if answer:
                        self.load_model(save_data['model_name'])
                    else:
                        return

            # 加载数据
            self.image_paths = save_data['image_paths']
            self.image_features = save_data['image_features'].astype(np.float32)  # 转换回float32以提高精度

            # 重建KNN模型
            self.knn_model = NearestNeighbors(n_neighbors=min(10, len(self.image_paths)), metric='cosine')
            self.knn_model.fit(self.image_features)

            self.status_var.set(f"特征加载完成，共 {len(self.image_paths)} 张图片 (模型: {self.current_model_name})")

        except Exception as e:
            messagebox.showerror("错误", f"加载特征失败: {str(e)}")
            self.status_var.set("特征加载失败")

    def search_images(self):
        """根据文本搜索图片"""
        if not self.knn_model:
            messagebox.showwarning("警告", "请先加载图片库或特征文件")
            return

        query_text = self.search_entry.get().strip()
        if not query_text:
            messagebox.showwarning("警告", "请输入搜索文本")
            return

        self.status_var.set("正在搜索...")

        try:
            # 在新线程中处理
            threading.Thread(target=self.process_search, args=(query_text,), daemon=True).start()
        except Exception as e:
            self.status_var.set(f"搜索错误: {str(e)}")

    def process_search(self, query_text):
        """改进的搜索处理流程"""
        try:
            # 文本预处理
            processed_text = self.preprocess_text(query_text)

            with torch.no_grad():
                # 使用混合精度加速
                with autocast():
                    text_input = clip.tokenize([processed_text]).to(self.device)
                    text_features = self.model.encode_text(text_input)

                # 多次编码取平均
                text_features_list = []
                for _ in range(3):  # 3次编码减少随机性
                    with autocast():
                        text_features_list.append(self.model.encode_text(text_input))
                text_features = torch.mean(torch.stack(text_features_list), dim=0)

            # 特征归一化
            text_features = text_features.cpu().numpy()
            text_features = text_features / np.linalg.norm(text_features)

            # 混合搜索策略
            text_sim = text_features @ self.image_features.T
            image_sim = self.image_features @ self.image_features.T
            combined_scores = 0.7 * text_sim + 0.3 * np.mean(image_sim, axis=1)

            # 获取Top20结果
            self.current_result_indices = np.argsort(-combined_scores[0])[:20]  # 修改：保存到成员变量

            # 显示结果
            self.result_listbox.delete(0, tk.END)
            for i, idx in enumerate(self.current_result_indices):  # 修改：使用enumerate
                img_name = os.path.basename(self.image_paths[idx])
                similarity = combined_scores[0][idx]
                self.result_listbox.insert(tk.END, f"{img_name} (相似度: {similarity:.3f})")
                # 根据相似度设置颜色
                color = "#%02x%02x%02x" % (
                    int(255 * (1 - similarity)),
                    int(255 * similarity),
                    128
                )
                self.result_listbox.itemconfig(tk.END, {'bg': color})


        except Exception as e:
            self.status_var.set(f"搜索错误: {str(e)}")

    def show_selected_image(self, event):
        """显示选中的图片 - 修复版"""
        selection = self.result_listbox.curselection()
        if not selection or self.current_result_indices is None:
            return

        listbox_idx = selection[0]
        if listbox_idx >= len(self.current_result_indices):
            return

        # 使用保存的索引获取正确的图片路径
        image_idx = self.current_result_indices[listbox_idx]
        if image_idx >= len(self.image_paths):
            messagebox.showerror("错误", "图片索引超出范围")
            return

        image_path = self.image_paths[image_idx]

        try:
            # 确保路径存在
            if not os.path.exists(image_path):
                messagebox.showerror("错误", f"图片文件不存在: {image_path}")
                return

            # 加载图片
            image = Image.open(image_path)

            # 获取显示区域实际大小（减去边距）
            display_width = self.image_frame.winfo_width() - 20
            display_height = self.image_frame.winfo_height() - 20

            # 计算保持宽高比的缩放比例
            img_width, img_height = image.size
            ratio = min(display_width / img_width, display_height / img_height)
            new_size = (int(img_width * ratio), int(img_height * ratio))

            # 高质量缩放
            image = image.resize(new_size, Image.Resampling.LANCZOS)

            # 创建PhotoImage并保持引用
            photo = ImageTk.PhotoImage(image)

            # 更新显示
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持引用避免被垃圾回收

            # 更新状态栏显示当前图片
            self.status_var.set(f"显示图片: {os.path.basename(image_path)}")

        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}\n路径: {image_path}")


if __name__ == "__main__":
    root = tk.Tk()


    # 在没有切换回调的情况下运行
    def no_switch():
        messagebox.showinfo("提示", "这是独立运行模式，切换功能不可用")


    app = CLIPImageSearchApp(root, switch_callback=no_switch)
    root.mainloop()