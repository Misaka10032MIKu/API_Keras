from keras.models import Model
from keras import layers
from keras import Input
import numpy as np

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(vocabulary_size, 256)(posts_input)
x = layers.Conv1D(filters=128, kernel_size=5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])
model.summary()

# 多重损失
# model.compile(loss={'age':'mse','income':'categorical_crossentropy','gender':'binary_crossentropy'},
#               optimizer='adam', metrics=['accuracy'])
# 损失加权
model.compile(
    loss={
        'age': 'mse',
        'income': 'categorical_crossentropy',
        'gender': 'binary_crossentropy'
    },
    optimizer='adam',
    # 为每个预测头（Head）精确指定评估指标
    metrics={
        'age': 'mae',                  # 年龄是回归任务，用平均绝对误差（MAE）更合理
        'income': ['accuracy'],        # 收入是多分类，用准确率
        'gender': ['accuracy']         # 性别是二分类，用准确率
    },
    loss_weights={
        'age': 0.25,
        'income': 1.0,
        'gender': 10.0
    }
)
num_samples = 1000
max_length = 360

print("--- 开始生成模拟数据 ---")
posts = np.random.randint(low=1, high=vocabulary_size, size=(num_samples, max_length), dtype=np.int32)
# 模拟真实文本：让部分样本的尾部变成 0（填充 Padding）
for i in range(num_samples):
    random_len = np.random.randint(50, max_length) # 假设每个帖子随机长度在 50~180 之间
    posts[i, random_len:] = 0  # : 表示 random 之后的全部元素

print(f"输入数据 (posts) 形状: {posts.shape} (样本数, 序列长度)")

# 3. 生成标签数据 (Targets)
# 任务 A: 年龄预测（回归问题 -> 1个连续值）
# 假设年龄分布在 18 到 70 岁之间
age_targets = np.random.uniform(18.0, 70.0, size=(num_samples, 1)).astype(np.float32)
print(f"年龄标签 (age_targets) 形状: {age_targets.shape}")

# 任务 B: 收入群体预测（多分类问题 -> 10维 One-Hot 编码）
# 先生成 0-9 的随机整数索引
income_indices = np.random.randint(0, num_income_groups, size=(num_samples,))
# 将其转换为 One-Hot 矩阵，对应 'categorical_crossentropy' 损失函数
income_targets = np.eye(num_income_groups)[income_indices].astype(np.float32)  # .eye 生成单位矩阵，[]内部进行矩阵切片
print(f"收入标签 (income_targets) 形状: {income_targets.shape}")

# 任务 C: 性别预测（二分类问题 -> 0或1，单列）
# 0 代表女性，1 代表男性
gender_targets = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
print(f"性别标签 (gender_targets) 形状: {gender_targets.shape}")

print("--- 数据生成完毕，可以送入 model.fit ---")

history = model.fit(posts,{'age':age_targets, 'income':income_targets, 'gender':gender_targets},
          epochs=10,batch_size=64,validation_split=0.2)

import matplotlib.pyplot as plt

epochs_range = range(1, len(history.history['loss']) + 1)

# 创建一个大图，包含 4 个子图（1个总损失 + 3个分支损失）
plt.figure(figsize=(15, 10))

# 子图 1：总损失变化 (Total Loss)
plt.subplot(2, 2, 1)
plt.plot(epochs_range, history.history['loss'], 'bo-', label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], 'ro-', label='Validation Loss')
plt.title('Total Combined Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 子图 2：年龄任务损失 (Age MSE Loss)
plt.subplot(2, 2, 2)
plt.plot(epochs_range, history.history['age_loss'], 'b--', label='Train Age Loss')
plt.plot(epochs_range, history.history['val_age_loss'], 'r--', label='Val Age Loss')
plt.title('Age Task Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 子图 3：收入任务损失 (Income Crossentropy Loss)
plt.subplot(2, 2, 3)
plt.plot(epochs_range, history.history['income_loss'], 'b--', label='Train Income Loss')
plt.plot(epochs_range, history.history['val_income_loss'], 'r--', label='Val Income Loss')
plt.title('Income Task Loss (Categorical CE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 子图 4：性别任务损失 (Gender Binary Crossentropy Loss)
plt.subplot(2, 2, 4)
plt.plot(epochs_range, history.history['gender_loss'], 'b--', label='Train Gender Loss')
plt.plot(epochs_range, history.history['val_gender_loss'], 'r--', label='Val Gender Loss')
plt.title('Gender Task Loss (Binary CE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 调整布局并展示
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)  # 保存为高清晰度图片，存放在项目根目录下
print("损失曲线图片已成功保存到项目根目录下的 loss_curve.png")
