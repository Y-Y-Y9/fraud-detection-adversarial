import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import random
import re
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 80)
print("对抗性数据改写在欺诈对话检测中的应用")
print("=" * 80)


# ==================== 1. 数据预处理 ====================
def preprocess_dialogue(text):
    """从对话内容中提取纯文本"""
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("音频内容：", "").replace("**", "")

    # 提取所有对话内容
    pattern = r'(?:left:|right:|LEFT:|RIGHT:)[ ]*(.*?)(?=(?:left:|right:|LEFT:|RIGHT:|$))'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        full_dialogue = " ".join([match.strip() for match in matches if match.strip()])
        full_dialogue = re.sub(r'\s+', ' ', full_dialogue).strip()
        return full_dialogue
    else:
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def load_and_preprocess_data(train_file, test_file):
    """加载和预处理数据"""
    print(f"加载训练集: {train_file}")
    train_df = pd.read_csv(train_file, encoding='utf-8')
    print(f"加载测试集: {test_file}")
    test_df = pd.read_csv(test_file, encoding='utf-8')

    print(f"\n训练集原始大小: {len(train_df)} 行")
    print(f"测试集原始大小: {len(test_df)} 行")

    # 处理缺失值
    train_df = train_df.dropna(subset=['is_fraud', 'specific_dialogue_content'])
    test_df = test_df.dropna(subset=['is_fraud', 'specific_dialogue_content'])

    # 转换标签
    def convert_label(x):
        if isinstance(x, bool):
            return 1 if x else 0
        elif isinstance(x, (int, float)):
            return int(float(x))
        elif isinstance(x, str):
            x = x.lower().strip()
            if x in ['true', 't', '1', 'yes', '是', '真']:
                return 1
            else:
                return 0
        else:
            return 0

    train_df['label'] = train_df['is_fraud'].apply(convert_label)
    test_df['label'] = test_df['is_fraud'].apply(convert_label)

    # 提取对话文本
    train_df['dialogue'] = train_df['specific_dialogue_content'].apply(preprocess_dialogue)
    test_df['dialogue'] = test_df['specific_dialogue_content'].apply(preprocess_dialogue)

    print(f"\n处理后训练集大小: {len(train_df)} 行")
    print(f"处理后测试集大小: {len(test_df)} 行")
    print(f"训练集标签分布: 欺诈={sum(train_df['label'])}, 正常={len(train_df) - sum(train_df['label'])}")
    print(f"测试集标签分布: 欺诈={sum(test_df['label'])}, 正常={len(test_df) - sum(test_df['label'])}")

    return train_df, test_df


# ==================== 2. 对抗性文本生成器 ====================
class SimpleAdversarialGenerator:
    def __init__(self):
        self.synonym_dict = {
            "客服": ["服务人员", "工作人员", "专员", "顾问"],
            "银行": ["金融机构", "储蓄所", "支行"],
            "贷款": ["借款", "信贷", "放款"],
            "退款": ["返款", "退还", "退回"],
            "密码": ["口令", "密钥", "PIN码"],
            "验证": ["核实", "确认", "审核"],
            "链接": ["网址", "URL", "网站"],
            "转账": ["汇款", "打款", "支付"],
            "账户": ["账号", "户头", "用户"],
            "信息": ["资料", "数据", "详情"],
            "提供": ["告知", "发送", "给出"],
        }

        # 欺诈关键词
        self.fraud_keywords = ["密码", "链接", "转账", "验证码", "身份证", "银行卡", "退款", "贷款"]

    def generate_adversarial(self, texts, labels, attack_type='synonym'):
        """生成对抗性样本"""
        adv_texts = []
        adv_labels = []

        for text, label in zip(texts, labels):
            if attack_type == 'synonym':
                # 同义词替换
                for old, news in self.synonym_dict.items():
                    if old in text and random.random() < 0.3:
                        text = text.replace(old, random.choice(news))
                adv_text = text
            elif attack_type == 'syntax':
                # 句式重组
                if random.random() < 0.5:
                    adv_text = text.replace('请', '').replace('您', '你')
                else:
                    adv_text = text + '，谢谢'
            elif attack_type == 'keyword':
                # 关键词替换（删除欺诈关键词）
                adv_text = text
                for keyword in self.fraud_keywords:
                    if random.random() < 0.4 and keyword in adv_text:
                        adv_text = adv_text.replace(keyword, '')
            else:
                # 混合攻击
                adv_text = text
                # 同义词替换
                for old, news in self.synonym_dict.items():
                    if old in adv_text and random.random() < 0.2:
                        adv_text = adv_text.replace(old, random.choice(news))
                # 句式重组
                if random.random() < 0.3:
                    adv_text = adv_text.replace('请', '')

            adv_texts.append(adv_text)
            adv_labels.append(label)

        return adv_texts, adv_labels


# ==================== 3. 简单文本分类模型 ====================
class SimpleTextClassifier:
    def __init__(self, model_type='lr'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=['的', '了', '在', '是', '我', '你', '他'])

        if model_type == 'lr':
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'svm':
            self.model = SVC(random_state=42, probability=True)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(random_state=42, n_estimators=100)

    def train(self, train_texts, train_labels):
        """训练模型"""
        print(f"训练{self.model_type}模型...")
        # 特征提取
        X_train = self.vectorizer.fit_transform(train_texts)
        # 训练模型
        self.model.fit(X_train, train_labels)
        return self

    def predict(self, texts):
        """预测"""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X), self.model.predict_proba(X)

    def evaluate(self, texts, labels, name="测试集"):
        """评估模型"""
        preds, probs = self.predict(texts)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        print(f"{name} - 准确率: {acc:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}")
        return acc, preds, probs


# ==================== 4. 主函数 ====================
def main():
    print("1. 数据加载与预处理...")

    try:
        train_df, test_df = load_and_preprocess_data("训练集结果.csv", "测试集结果.csv")
    except Exception as e:
        print(f"数据加载失败: {e}")

    # 准备数据
    train_texts = train_df['dialogue'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['dialogue'].tolist()
    test_labels = test_df['label'].tolist()

    print(f"\n2. 数据统计:")
    print(f"   训练集: {len(train_texts)} 个样本")
    print(f"   测试集: {len(test_texts)} 个样本")

    # 生成对抗性样本
    print("\n3. 生成对抗性样本...")
    generator = SimpleAdversarialGenerator()

    print("   a) 同义词替换攻击...")
    syn_texts, syn_labels = generator.generate_adversarial(test_texts, test_labels, 'synonym')

    print("   b) 句式重组攻击...")
    synx_texts, synx_labels = generator.generate_adversarial(test_texts, test_labels, 'syntax')

    print("   c) 关键词删除攻击...")
    key_texts, key_labels = generator.generate_adversarial(test_texts, test_labels, 'keyword')

    print("   d) 混合攻击...")
    mix_texts, mix_labels = generator.generate_adversarial(test_texts, test_labels, 'mixed')

    print("\n   对抗样本示例:")
    if len(test_texts) > 0:
        print(f"   原始: {test_texts[0][:80]}...")
        print(f"   同义词: {syn_texts[0][:80]}...")
        print(f"   句式重组: {synx_texts[0][:80]}...")
        print(f"   混合: {mix_texts[0][:80]}...")

    # 训练和评估多个模型
    print("\n4. 训练和评估模型...")

    results = []
    models = {
        '逻辑回归': 'lr',
        '支持向量机': 'svm',
        '随机森林': 'rf'
    }

    for model_name, model_type in models.items():
        print(f"\n{model_name}模型:")

        # 训练模型
        classifier = SimpleTextClassifier(model_type)
        classifier.train(train_texts, train_labels)

        # 评估不同测试集
        orig_acc, _, _ = classifier.evaluate(test_texts, test_labels, "原始测试集")
        syn_acc, _, _ = classifier.evaluate(syn_texts, syn_labels, "同义词替换")
        synx_acc, _, _ = classifier.evaluate(synx_texts, synx_labels, "句式重组")
        key_acc, _, _ = classifier.evaluate(key_texts, key_labels, "关键词删除")
        mix_acc, _, _ = classifier.evaluate(mix_texts, mix_labels, "混合攻击")

        # 保存结果
        results.append({
            '模型': model_name,
            '原始准确率': orig_acc,
            '同义词替换': syn_acc,
            '句式重组': synx_acc,
            '关键词删除': key_acc,
            '混合攻击': mix_acc,
            '最大下降': orig_acc - min(syn_acc, synx_acc, key_acc, mix_acc)
        })

    # 结果分析
    print("\n5. 结果分析:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string())

    # 找出表现最好的模型
    best_model_idx = results_df['原始准确率'].idxmax()
    best_model = results_df.loc[best_model_idx, '模型']
    best_orig_acc = results_df.loc[best_model_idx, '原始准确率']
    best_max_drop = results_df.loc[best_model_idx, '最大下降']

    print(f"\n   最佳模型: {best_model}")
    print(f"   原始准确率: {best_orig_acc:.4f}")
    print(f"   最大准确率下降: {best_max_drop:.4f}")

    # 可视化
    print("\n6. 生成可视化图表...")
    plt.figure(figsize=(14, 8))

    # 图1: 不同模型的性能对比
    plt.subplot(2, 2, 1)
    models_list = results_df['模型'].tolist()
    x = np.arange(len(models_list))
    width = 0.15

    plt.bar(x - 2 * width, results_df['原始准确率'], width, label='原始', color='blue')
    plt.bar(x - width, results_df['同义词替换'], width, label='同义词', color='red')
    plt.bar(x, results_df['句式重组'], width, label='句式重组', color='orange')
    plt.bar(x + width, results_df['混合攻击'], width, label='混合攻击', color='green')

    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.title('不同模型在不同攻击下的准确率')
    plt.xticks(x, models_list, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 图2: 准确率下降幅度
    plt.subplot(2, 2, 2)
    drops = []
    attack_types = ['同义词替换', '句式重组', '混合攻击']
    colors = ['red', 'orange', 'green']

    for i, attack in enumerate(attack_types):
        drop = results_df['原始准确率'] - results_df[attack]
        plt.bar(x + (i - 1) * width, drop, width, label=attack, color=colors[i], alpha=0.7)

    plt.xlabel('模型')
    plt.ylabel('准确率下降')
    plt.title('不同攻击方式下的准确率下降')
    plt.xticks(x, models_list, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 图3: 最佳模型的详细分析
    plt.subplot(2, 2, 3)
    best_data = results_df[results_df['模型'] == best_model].iloc[0]

    attack_types_full = ['原始', '同义词替换', '句式重组', '关键词删除', '混合攻击']
    acc_values = [
        best_data['原始准确率'],
        best_data['同义词替换'],
        best_data['句式重组'],
        best_data['关键词删除'],
        best_data['混合攻击']
    ]

    colors_full = ['blue', 'red', 'orange', 'purple', 'green']
    bars = plt.bar(attack_types_full, acc_values, color=colors_full)
    plt.xlabel('测试集类型')
    plt.ylabel('准确率')
    plt.title(f'{best_model}模型在不同攻击下的准确率')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, acc in zip(bars, acc_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

    # 图4: 对抗性攻击效果总结
    plt.subplot(2, 2, 4)

    # 计算平均下降
    avg_drops = []
    for attack in ['同义词替换', '句式重组', '混合攻击']:
        avg_drop = (results_df['原始准确率'] - results_df[attack]).mean()
        avg_drops.append(avg_drop)

    plt.bar(['同义词', '句式', '混合'], avg_drops, color=['red', 'orange', 'green'])
    plt.xlabel('攻击类型')
    plt.ylabel('平均准确率下降')
    plt.title('不同攻击方式的平均效果')
    plt.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, drop in enumerate(avg_drops):
        plt.text(i, drop + 0.01, f'{drop:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('fraud_detection_simple_results.png', dpi=300, bbox_inches='tight')
    print("图表已保存为: fraud_detection_simple_results.png")

    # 保存结果
    results_df.to_csv('experiment_simple_results.csv', index=False, encoding='utf-8')
    print("结果已保存为: experiment_simple_results.csv")

    # 总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)

    print(f"1. 最佳模型: {best_model}")
    print(f"2. 原始测试集准确率: {best_orig_acc:.4f}")
    print(f"3. 对抗性攻击最大准确率下降: {best_max_drop:.4f}")

    # 判断脆弱性程度
    if best_max_drop > 0.2:
        vulnerability = "非常脆弱"
    elif best_max_drop > 0.1:
        vulnerability = "比较脆弱"
    elif best_max_drop > 0.05:
        vulnerability = "有一定脆弱性"
    else:
        vulnerability = "相对鲁棒"

    print(f"4. 模型鲁棒性评估: {vulnerability}")

    # 最有效的攻击方式
    attack_results = {
        '同义词替换': best_data['原始准确率'] - best_data['同义词替换'],
        '句式重组': best_data['原始准确率'] - best_data['句式重组'],
        '混合攻击': best_data['原始准确率'] - best_data['混合攻击']
    }
    most_effective = max(attack_results, key=attack_results.get)
    print(f"5. 最有效的攻击方式: {most_effective} (下降: {attack_results[most_effective]:.4f})")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()