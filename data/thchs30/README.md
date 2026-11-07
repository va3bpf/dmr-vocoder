# THCHS-30 中文测试句

来自清华大学 THCHS-30 普通话语音库的测试集，取自 Hugging Face 上的
<https://huggingface.co/datasets/FluidInference/THCHS-30-tests>（Apache License 2.0）。

- 10 个说话人（D4、D6、D7、D8、D11、D12、D13、D21、D31、D32）各一句，都是第 750 句：
  “东北军的一些爱国将士马占山、李杜、唐聚伍、苏炳艾、邓铁梅”。
- 16 kHz、16 bit、单声道 WAV，每句约 10 秒；编码前要降到 8 kHz。
- `metadata.jsonl`：整个测试集（2495 句）的文件名和文本；`pick.txt`：这里挑的 10 句。

THCHS-30: Dong Wang, Xuewei Zhang, Zhiyong Zhang, "THCHS-30: A Free Chinese Speech Corpus", 2015.
