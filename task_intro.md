补充一点：我们的回归任务不只有 `LIR` 和 `Jaccard`，还包括数据集中的 **sentence jaccard** 指标。因此，所有外部方法在适配后都需要同时支持 **3 个回归目标**：

* `LIR`
* `Jaccard`
* `Sentence Jaccard`

请按这个新定义执行，下面是更新后的关键段落，可直接替换原指令中的对应部分。

---

## 当前数据字段说明

当前项目实际使用的数据文件可以是显式划分的 `train.jsonl / test.jsonl / val.jsonl`，例如 `mydata/train.jsonl` 与 `mydata/test.jsonl`。这类数据的核心字段包括：

* `id`
* `source_dataset`
* `source_model`
* `source_domain`
* `original_text`
* `mixed_text`
* `rewritten_text`
* `target_ai_ratio`
* `n_sentences`
* `sentence_labels`
* `ai_sentences_original`
* `ai_sentences_rewritten`
* `mixing_mode`
* `rewrite_info`

其中需要特别注意：

* `mixed_text` 是当前用于分类与回归任务的主输入文本。
* `original_text` 是原始参考文本。
* `rewritten_text` 是仅将被选中的 AI 句子进一步 humanize 后得到的版本，可作为辅助分析字段，但不是必须输入。
* `sentence_labels` 是逐句标签，`1` 表示该句属于 AI/rewrite 句子，`0` 表示保留原句。
* `target_ai_ratio` 是六分类任务对应的 gold ratio bin 来源。
* `ai_sentences_original` 与 `ai_sentences_rewritten` 提供了被替换句子的前后对应关系，可用于辅助特征构造或误差分析。
* `rewrite_info` 记录 rewriter、prompt、时间戳等元信息，可用于实验说明，但通常不应直接作为监督标签输入模型。

---

## 更新版任务定义

你的任务是将一个外部 AI-text detection 方法迁移到我们的数据集与任务设定中。请不要仅复现原论文/原仓库默认任务，而是统一改造成适用于我们项目的两个任务：

### 1. 目标任务

#### 任务 A：六分类任务

将每篇文档预测为以下 6 个 AI ratio bin 之一：

* 0%
* 20%
* 40%
* 60%
* 80%
* 100%

#### 任务 B：三回归任务

在同一批样本上输出三个连续回归结果：

* LIR 回归值
* Jaccard 回归值
* Sentence Jaccard 回归值

如果原方法只能输出单个分数，请尽量基于其原始输出、中间特征、logits、相似度、置信度或可学习头部扩展为支持这三个回归目标；如果确实无法直接支持，请明确说明限制，并给出你采用的最合理改造方案。

对于当前 `mydata` 这类字段格式，需要明确区分“直接提供的标签”和“需要派生的标签”：

* 六分类标签：由 `target_ai_ratio` 直接映射到 `0/20/40/60/80/100` 六类。
* `LIR`：如果数据中没有现成字段，可由 `sentence_labels` 的均值派生；必要时可记为 `sum(sentence_labels) / len(sentence_labels)`。
* `Jaccard`：如果数据中没有现成字段，可由 `original_text` 与 `mixed_text` 的 token-level Jaccard distance 派生。
* `Sentence Jaccard`：如果数据中没有现成字段，可由 `original_text` 与 `mixed_text` 的 sentence-level Jaccard distance 派生。

也就是说，当前数据集中并不一定直接给出 `LIR / Jaccard / Sentence Jaccard` 三个数值字段；方法迁移时需要把这三个监督目标的构造过程明确写进实现说明中。

---

## 更新版输出格式要求

每个样本最终至少输出：

* `pred_class_6way`
* `prob_6way` 或 6 类 logits / probabilities
* `pred_lir`
* `pred_jaccard`
* `pred_sentence_jaccard`

同时保留：

* 样本 id
* gold label（六分类标签、LIR、Jaccard、Sentence Jaccard）
* 方法名
* split 名称
* 如标签为派生得到，建议同时保留 target derivation note，例如 `dataset`、`derived_from_sentence_labels`、`derived_from_token_sets`、`derived_from_sentence_sets`

---

## 更新版回归任务改造要求

在六分类之外，还要支持三个回归目标：

* `LIR`
* `Jaccard`
* `Sentence Jaccard`

要求如下：

1. **尽量共享 backbone**

   * 如果方法本身是可训练神经模型，优先在共享编码器后增加三个独立回归头。
   * 可采用 multi-task learning。

2. **输出形式**

   * `pred_lir`: 连续值
   * `pred_jaccard`: 连续值
   * `pred_sentence_jaccard`: 连续值

3. **损失函数建议**
   可根据方法类型选择：

   * `MSELoss`
   * `L1Loss`
   * 或加权组合：

```text
L_total = L_cls + lambda1 * L_lir + lambda2 * L_jaccard + lambda3 * L_sentence_jaccard
```

4. **如果原方法不可训练**

   * 尝试基于其中间分数构造可学习回归器，例如：

     * 线性回归
     * MLP regression head
     * logistic/ordinal mapping
   * 输入可以是原方法输出的：

     * logit
     * similarity score
     * perplexity-like score
     * rank features
     * retrieval features
    * token statistics
    * 如果只能构造弱映射，也要保留并说明它是“adapted regression head”，不是原方法原生能力。

---

## 更新版训练监控要求

训练型方法需要接入 `Weights & Biases (wandb)` 做实时监测，至少覆盖以下内容：

* `wandb.init(...)`
* 记录主要超参数到 `config`
* 训练过程中按 step 或 batch 实时 `wandb.log(...)`
* 每个 epoch 记录训练集与验证集指标
* 训练结束后记录测试集最终指标
* 训练结束时调用 `wandb.finish()`

建议至少监控：

* `train/loss`
* `valid/loss`
* `train/cls_loss`
* `train/lir_loss`
* `train/jaccard_loss`
* `train/sentence_jaccard_loss`
* `valid/cls_loss`
* `valid/lir_loss`
* `valid/jaccard_loss`
* `valid/sentence_jaccard_loss`
* `best_valid_loss`
* 测试集 `Macro-F1`
* 测试集 `AUROC`
* 测试集 `MAE/MSE` for `LIR`
* 测试集 `MAE/MSE` for `Jaccard`
* 测试集 `MAE/MSE` for `Sentence Jaccard`

如果项目环境不便联网，至少要支持 `wandb offline` 模式；如果默认不开启，也要提供显式命令行开关，例如 `--use_wandb`。

---

## 更新版评测指标要求

请严格按下面输出结果。

### 六分类任务指标

必须报告：

* 每个类别的 F1

  * F1@0%
  * F1@20%
  * F1@40%
  * F1@60%
  * F1@80%
  * F1@100%

* 总 F1

  * 默认使用 **macro-F1**

* 总 AUROC

  * 对六分类任务使用 **multi-class AUROC**
  * 推荐统一使用 **macro one-vs-rest AUROC**

### 三回归任务指标

必须分别报告：

* `MAE(LIR)`
* `MSE(LIR)`
* `MAE(Jaccard)`
* `MSE(Jaccard)`
* `MAE(Sentence Jaccard)`
* `MSE(Sentence Jaccard)`

---

## 更新版结果表格式

| Method | F1@0 | F1@20 | F1@40 | F1@60 | F1@80 | F1@100 | Macro-F1 | AUROC | MAE(LIR) | MSE(LIR) | MAE(Jaccard) | MSE(Jaccard) | MAE(Sent-Jaccard) | MSE(Sent-Jaccard) |
| ------ | ---- | ----- | ----- | ----- | ----- | ------ | -------- | ----- | -------- | -------- | ------------ | ------------ | ----------------- | ----------------- |

---

## 更新版最终交付要求

结果文件中必须包含：

* 每样本六分类预测
* 每样本 `LIR` 预测
* 每样本 `Jaccard` 预测
* 每样本 `Sentence Jaccard` 预测
* 汇总指标表
* 可直接粘贴到论文/slide 的表格

如果使用的是 `mydata` 这类没有直接提供全部回归标签的数据格式，还必须说明：

* 训练集、验证集、测试集分别来自哪些 jsonl 文件
* `LIR` 的构造规则
* `Jaccard` 的构造规则
* `Sentence Jaccard` 的构造规则
* 每条样本是否需要输出 `target_notes` 或等价字段，标明目标值是原始提供还是派生得到

如果方法是可训练的，还应补充：

* `wandb` 运行配置说明
* 训练监控日志对应的项目名 / run 名
* 如使用 `offline` 模式，需说明本地日志目录或同步方式

---

