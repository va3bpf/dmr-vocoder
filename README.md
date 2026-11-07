# DMR 声码器（AMBE+2 兼容）

[English](README.en.md) · 用在 Baofeng DM-32UV 的开源固件 [OpenDMR](https://github.com/va3bpf/OpenDMR) 里（本仓库是它的子模块）。

DMR 用的 AMBE+2 半速率声码器（3600x2450：每 20 ms 一帧，49 bit 参数 + 23 bit 纠错 = 72 bit）的编码器和解码器，
自己实现，和 DVSI 的声码器互通。同一套算法有三份实现：

| 实现 | 位置 | 用途 |
|---|---|---|
| Python | `py/ambe_encoder.py`、`py/ambe_analysis.py`、`py/ambe_decoder.py` | 研究和调参的基准，参数都是模块常量 |
| C 双精度 | `c/ref/` | 和 Python 逐位一致，用来检查定点版的误差 |
| C 定点 | `c/ambe/` | 嵌入式用，没有 FPU 也能跑：整数和查表，每帧只有几次 float |

**现状**：C 定点版已在 DM-32 电台（CK803S，168 MHz）上实际收发，空中和商用 DMR 电台互通。
每帧解码 2.40 ms、编码 13.1 ms，占 RAM 53.4 KB。电台上的输出和电脑上的 `bin/ambe_cli.exe` 逐位一致。
六句中文（zh6）经 DVSI 解码器的 PESQ：我们的编码器 3.13，DVSI 自己的编码器 2.97。

- 算法：`docs/algorithm.md`
- 研究过程、试过的做法和数字：`docs/notes.md`

## 目录

| 路径 | 内容 |
|---|---|
| `c/ambe/` | 定点声码器：`ambe.h` 是接口，`ambe_enc.c` / `ambe_dec.c` / `ambe_parms.c` / `ambe_frame.c`，表 `ambe_tables.c`（`py/gen_c_tables.py` 生成） |
| `c/ref/` | 双精度参考版，同样的接口 |
| `c/test/ambe_cli.c` | 命令行：`ambe_cli enc in.raw out.bit` / `ambe_cli dec in.bit out.raw` |
| `c/analysis.c` | 给 Python 编码器加速的谐波拟合（ctypes 调用，快约 15 倍） |
| `c/pesq_main.c` | PESQ 命令行 |
| `c/stack_tramp.S` | 电台测试用的换栈跳板（汇编结果嵌在 `py/radio_link.py` 里） |
| `c/build.sh` | 把上面的编成 `bin/` 里的电脑程序 |
| `py/` | Python 实现和全部测试脚本（下面逐个说明） |
| `data/thchs30/` | 测试语音：THCHS-30 十个人读同一句中文（说明见其中的 README） |
| `data/capture/` | `hotspot_capture.py` 空中录到的 DMR 语音帧，**不进 git** |
| `data/dvsi/` | DVSI 官方测试向量，**不进 git**（见“准备”） |
| `ref/` | 参考源码（mbelib、dsd、OP25、PESQ），**不进 git**，`fetch_refs.sh` 拉取 |
| `bin/` | `c/build.sh` 的输出，**不进 git** |
| `out/` | 测试的输出（图、WAV、缓存），**不进 git** |
| `tools/` | 在 WSL 里编译、运行 md380-emu（第二个 DVSI 参考） |

## 准备

只用电脑的测试需要 1～4，用 md380 的需要 5，电台测试另见下面“二”。

1. **Python 3** 和包：`pip install -r requirements.txt`（numpy、scipy、matplotlib、pystoi；电台测试要 pyserial）。
2. **参考源码**：`sh fetch_refs.sh` → `ref/`。`c/build.sh` 要其中的 PESQ；`py/gen_tables.py` 要 mbelib。
3. **DVSI 测试向量**：DVSI 公开的 AMBE-3000 测试包 <https://www.dvsinc.com/get-hdk_3000/hdkiso/tv.zip>
   （约 520 MB，页面 <https://www.dvsinc.com/dlapps/appsoft.shtml>）。把根目录的 `dam.pcm`、`clean.pcm`、`noisy.pcm`
   和 `r33/` 目录解压到 `data/dvsi/`。r33 就是 DMR 用的 3600x2450：
   - `dam.pcm` 等：原始输入，8 kHz 16 bit 小端；
   - `r33/<名>.bit`：DVSI 编码器的输出，每帧 9 字节；`r33/<名>.pcm`：DVSI 解码这些比特的结果，是解码器的标准答案；
   - `_e1` / `_e2` / `_e5` / `_e10_hd`：误码率 1～10 % 的硬判决比特，`_sd` 软判决，`_dtx` 带静音压缩。
4. **电脑上的 C 程序**：`sh c/build.sh` → `bin/pesq_nb.exe`、`bin/vocoder_c.dll`、`bin/ambe_cli.exe`、`bin/ambe_ref.exe`。
   用 PATH 里的 gcc（MinGW-w64）；脚本会先把 `../toolchain/winlibs/mingw64/bin` 加到 PATH 前面，有就用那个。
5. **md380-emu**（可选，第二个 DVSI 参考，比用电台快得多）：MD-380 固件里的 DVSI 声码器在 qemu 下运行。
   在 WSL（Ubuntu 22.04）里：`wsl -u root -- bash /mnt/<仓库路径>/tools/build_md380emu.sh`，
   需要的 apt 包写在脚本开头。它和 DM-32 原厂的编解码器逐位相同，所以“原厂”的对比都可以用它代替电台。

测试语音 zh6（`out/zh6_input.wav`：THCHS-30 的 D4、D6、D7、D8、D11、D12 六句，带通 300～3400 Hz、电平 RMS 3000，
58 秒）由 `python py/zh6_input.py` 生成；用到它的脚本发现没有会自动生成。

## 测试

所有脚本都在 `py/` 里运行（`cd py`），输出写到 `out/`。

### 一、只用电脑

| 命令 | 测什么 | 需要（见“准备”） |
|---|---|---|
| `python ambe_frame.py` | 帧层自测：72 bit ↔ 49 bit 往返、Golay 纠错、DVSI 向量重编码 | 3 |
| `python ambe_encoder.py` | 量化往返：DVSI 比特解出参数再量化，参数应完全相同 | 3 |
| `python ambe_tone.py` | 单音帧：NXDN 公开的 1031 Hz 帧解析和重新生成 | — |
| `python tone_test.py` | 解码器的单音：1031 Hz 测试音、单音扫频、DTMF 的频率、电平、SINAD、THD；图 `out/tone_test.png` | — |
| `python evaluate.py [名 ...]` | 解码器对 DVSI 解码的频谱距离 LSD，按浊音 / 混合 / 清音分组（默认 dam clean noisy） | 3 |
| `python compare.py [名]` | 解码一个 DVSI 向量，画输入、DVSI、mbelib 原样、我们的波形 / 语谱图 / 频谱，导出 WAV | 3 |
| `python decode.py x.bit ...` | 解码任意 `.bit`（比如空中录的），WAV 和图（基频、清浊音、每帧纠错数） | — |
| `python encoder_test.py [名 ...]` | 编码器对 DVSI 编码器：逐帧参数、端到端 LSD、对原始输入的 PESQ / STOI | 3、4 |
| `python md380_eval.py [模块.常量=值 ...]` | **调参的主要工具**：编码器经 md380 解码、md380 的比特经我们解码，对原声的 PESQ / STOI；可临时改任何模块常量 | 4、5 |
| `python c_eval.py [enc] [dec]` | C 定点版对 Python 版：编码逐字段一致率，解码的 PESQ（C 定点 / C 双精度 / Python） | 4、5 |

`md380_eval.py` 第一次运行时用 md380 编码、解码，结果缓存在 `out/md380_eval_ref.json` 和 `out/md380_<名>.json`；
`c_eval.py dec` 要用这些缓存，所以先跑一次 `md380_eval.py`。例：

```
python md380_eval.py                                     # 当前默认值
python md380_eval.py ambe_analysis.VOICED_ERR=0.5        # 试一个参数，看 PESQ 相对 md380 的变化
```

### 二、用电台（DM-32，写频线）

把 DM-32 当作“装着原厂 DVSI 编解码器和我们 C 定点版的设备”来用，通过写频线的调试命令写内存、调用函数。条件：

- 电台装 OpenDMR 固件，并且这个仓库是 OpenDMR 的子模块：脚本从 `../src/build/opendmr.elf` 取函数地址
  （**ELF 必须和电台里的固件是同一次编译**），用 `../toolchain/` 里的 `csky-elfabiv2-nm`。
- 电台开机、在 **FM 信道**空闲、不在热点模式，写频线在 COM6（各脚本有 `--port`）。DMR 信道上电台自己的
  音频会同时用声码器，结果不对。
- pyserial。

| 命令 | 测什么 |
|---|---|
| `python vocoder_bench.py exact [帧数]` | **C 定点版在电台上和电脑上逐位一致**：编码、解码各逐帧比较（默认 zh6 第 100 帧起 200 帧）；全部一致时退出码 0 |
| `python vocoder_bench.py time` | 电台上每帧解码 / 编码的毫秒数、编码各步骤的时间、软件浮点和整数运算的周期数 |

`time` 的正常结果：解码约 2.4 ms / 帧，编码约 13 ms / 帧（实时上限 20 ms）。

### 三、空中（两台电台）

DM-32 进热点模式（OpenDMR 固件），电脑通过写频线把它当 MMDVM 用：

| 命令 | 做什么 |
|---|---|
| `python hotspot_capture.py [名] [--freq 433450000] [--cc 1]` | 录另一台 DMR 电台发来的语音，每次讲话存一个 `data/capture/<名>_<n>.bit`，再用 `decode.py` 解码 |
| `python hotspot_tx.py a.wav ... [--tg 9] [--src 4600001]` | 用我们的编码器编码 wav（或直接发 `.bit`），通过热点发成组呼，用别的电台收听；`--save x.bit` 同时存下比特 |

频率和色码要和另一台电台一致。PESQ 对“机器人声”等问题不敏感，编码器的改动最后以空中试听为准。

## 常用流程

- **调 Python 编码器 / 解码器的参数**：`md380_eval.py 模块.常量=值`，看 PESQ 相对 md380 的变化；
  好的再用 `hotspot_tx.py` 空中试听。
- **改了 C 定点版**：`sh c/build.sh`，`python c_eval.py`（和 Python 版一致率、PESQ 不能变差）；
  然后编译、写入电台固件，`python vocoder_bench.py exact`（必须全部一致）和 `time`（不能超过 20 ms）。
- **改了 Python 版的算法**：C 双精度版 `c/ref/` 要跟着改到逐位一致，再改定点版。
- **码表**：`python gen_tables.py`（mbelib 头文件 → `py/mbe_tables.py`），`python gen_c_tables.py`（→ `c/ambe/ambe_tables.c`）。

## 文件格式

| 扩展名 | 内容 |
|---|---|
| `.pcm` / `.raw` | 8 kHz、16 bit、小端、单声道，没有文件头 |
| `.wav` | 同上加 WAV 头；脚本读别的采样率时会重采样到 8 kHz |
| `.bit` | 每帧 9 字节 = 72 bit 信道比特（DMR 交织顺序，字节内高位在前），和 DVSI 的 `.bit` 相同 |
| `.amb` | DSD / md380-emu 的格式：文件头 `.amb`，每帧 8 字节（状态字节 + 49 bit，高位在前） |

一个 DMR 语音突发带 3 帧 = 27 字节（MMDVM 也是这个格式）。帧层细节见 `docs/algorithm.md` 第 1 节。

## 许可和专利

- 本仓库：BSD 式许可，第 4 条**商用须经 Junqiang Shi（VA3BPF）书面同意**，全文见 [LICENSE](LICENSE)。
- 解码器从 mbelib 移植（ISC 许可，许可声明见 LICENSE）；码表 `py/mbe_tables.py`、`c/*/ambe_tables.c` 由 mbelib 的头文件生成。
- OP25 的编码器是 GPL v3，只用来理解算法，没有搬代码。
- `data/thchs30/`：THCHS-30 的测试句，Apache License 2.0（见其中的 README）。
- AMBE+2 是 DVSI 的专利技术，部分专利可能还没到期，公开发布或商用前要查清楚。
