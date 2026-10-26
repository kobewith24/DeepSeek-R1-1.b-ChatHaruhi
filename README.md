# 鍩轰簬 DeepSeek-R1-1.5B 鐨勪簩娆″厓瑙掕壊鎵紨鍨傚煙楂樻晥寰皟

> 浠跨収銆屽熀浜?Qwen-3 鐨勫瀭鍩熷ぇ妯″瀷楂樻晥寰皟銆嶉」鐩€濊矾锛屽湪 **Windows 鍗曟満锛圧TX 4060 Laptop 8GB锛?* 涓婂鐜板瀭鍩熷ぇ妯″瀷楂樻晥寰皟鍏ㄦ祦绋嬶細
> 鏁版嵁鏋勫缓 鈫?QLoRA 楂樻晥寰皟 鈫?鏉冮噸鍚堝苟 鈫?Ollama 閮ㄧ讲 鈫?瀵规瘮璇勬祴涓庡帇娴嬨€?

## 椤圭洰绠€浠?

闈㈠悜**浜屾鍏冭鑹叉壆婕?*鍨傚煙鐨?DeepSeek-R1-Distill-Qwen-1.5B 楂樻晥寰皟銆傚湪鎻愬崌妯″瀷瑙掕壊鎵紨鑳藉姏锛堣姘斻€佹€ф牸銆佸彛鐧栬繕鍘燂級鐨勫悓鏃讹紝閫氳繃娣峰悎閫氱敤瀵硅瘽鏁版嵁淇濈暀鍏堕€氱敤鑳藉姏锛岄伩鍏嶇伨闅炬€ч仐蹇樸€?

## 鎶€鏈爤

- **寰皟**锛歅ython銆丳EFT锛圠oRA/QLoRA锛夈€乀RL锛圫FTTrainer锛夈€乥itsandbytes锛?bit NF4 鍔ㄦ€侀噺鍖栵級
- **閮ㄧ讲**锛歄llama锛圤penAI 鍏煎 API锛宻afetensors 鐩存帴瀵煎叆锛?
- **璇勬祴**锛氳嚜鐮斿姣旇瘎娴嬭剼鏈?+ 鍘嬫祴鑴氭湰锛堝悶鍚?tok/s锛?

> 涓庡師椤圭洰锛圦wen3-14B + Unsloth + vLLM + EvalScope锛夌殑瀵圭収锛歎nsloth 鈫?PEFT/TRL锛圲nsloth 涓嶆敮鎸?Windows 鍘熺敓锛夛紱vLLM 鈫?Ollama锛汦valScope 鈫?鑷爺璇勬祴鑴氭湰銆俀LoRA 鏍稿績鎶€鏈矾绾夸笉鍙樸€?

## 鐩綍缁撴瀯

```
DeepSeek-R1-FT/
鈹溾攢 model/                          # 鍩哄骇妯″瀷 DeepSeek-R1-Distill-Qwen-1.5B锛圡odelScope 涓嬭浇锛?
鈹溾攢 data/
鈹? 鈹溾攢 build_dataset.py             # 鏁版嵁闆嗘瀯寤猴細ChatHaruhi-54K + FineTome-100k锛?5/75 娣峰悎
鈹? 鈹斺攢 cleaned_roleplay_dataset/    # 娣峰悎鍚庣殑璁粌鏁版嵁闆嗭紙datasets 鏍煎紡锛?
鈹溾攢 train/
鈹? 鈹溾攢 finetune.py                  # QLoRA 寰皟锛?bit 閲忓寲 + LoRA r=32锛屽惈鏄惧瓨/loss 璁板綍锛?
鈹? 鈹溾攢 merge.py                     # LoRA 鍚堝苟瀵煎嚭 fp16
鈹? 鈹溾攢 training_log.json            # 璁粌鏃ュ織锛坙oss 鏇茬嚎銆佹樉瀛樺崰鐢級
鈹? 鈹斺攢 checkpoints/                 # 璁粌杩囩▼鏂囦欢
鈹溾攢 weights/
鈹? 鈹溾攢 lora_adapter/                # LoRA adapter 鏉冮噸
鈹? 鈹斺攢 DeepSeek-R1-1.5B-roleplay-fp16/  # 鍚堝苟鍚庣殑 fp16 瀹屾暣妯″瀷
鈹溾攢 deploy/
鈹? 鈹斺攢 Modelfile                    # Ollama 瀵煎叆閰嶇疆锛堝惈 R1 瀵硅瘽妯℃澘锛?
鈹溾攢 eval/
鈹? 鈹溾攢 eval_compare.py              # 寰皟鍓嶅悗瀵规瘮璇勬祴锛堣鑹叉壆婕?+ 閫氱敤闃查仐蹇橈級
鈹? 鈹溾攢 benchmark.py                 # 20 娆¤姹傚帇娴嬶紙鎴愬姛鐜?+ tok/s锛?
鈹? 鈹溾攢 make_dashboard.py            # EvalScope 鎴愮哗浠〃鐩橈紙鐜姸鍥撅級鐢熸垚
鈹? 鈹溾攢 evalscope/                   # EvalScope 璇勬祴鍘熷鎶ュ憡涓庨娴?
鈹? 鈹斺攢 reports/                     # 瀵规瘮鎶ュ憡銆佸帇娴嬬粨鏋溿€乨ashboard.png
鈹斺攢 requirements.txt
```

## 澶嶇幇姝ラ

```powershell
# 1. 鍒涘缓鐜锛圥ython 3.12锛?
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# torch 闇€ CUDA 鐗堬細
.venv\Scripts\python -m pip install --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128 torch

# 2. 涓嬭浇鍩哄骇妯″瀷锛堢害 3.4GB锛孧odelScope 鍥藉唴婧愶級
.venv\Scripts\python -c "from modelscope import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', local_dir=r'model/DeepSeek-R1-Distill-Qwen-1.5B')"

# 3. 鏋勫缓娣峰悎鏁版嵁闆嗭紙25% 瑙掕壊鎵紨 + 75% 閫氱敤瀵硅瘽锛?
.venv\Scripts\python data\build_dataset.py

# 4. QLoRA 寰皟
.venv\Scripts\python train\finetune.py

# 5. 鍚堝苟鏉冮噸
.venv\Scripts\python train\merge.py

# 6. 瀵煎叆 Ollama
ollama create deepseek-r1-roleplay -f deploy/Modelfile

# 7. 瀵规瘮璇勬祴 + 鍘嬫祴
.venv\Scripts\python eval\eval_compare.py
.venv\Scripts\python eval\benchmark.py
```

## 瀹為獙缁撴灉

<!-- 璁粌瀹屾垚鍚庡～鍐?-->

### 璁粌閰嶇疆涓庣粨鏋?

| 椤圭洰 | 鏁板€?|
|---|---|
| 璁粌鏍锋湰 | 10,000 鏉★紙2,500 瑙掕壊鎵紨 + 7,500 閫氱敤瀵硅瘽锛?5%/75%锛?|
| 璁粌姝ユ暟 | 625 姝ワ紙1 epoch锛宔ffective batch = 2 脳 8锛?|
| 閲忓寲鏂瑰紡 | 4bit NF4锛坆itsandbytes锛夛紝LoRA r=32 |
| 璁粌鏄惧瓨宄板€?| **6.35 GB / 8 GB**锛圧TX 4060 Laptop锛屽崟鍗″彲璁級 |
| 璁粌鏃堕暱 | 绾?3 灏忔椂 59 鍒嗭紙4bit QLoRA锛?048 搴忓垪闀垮害锛?|
| Loss | 1.74 鈫?0.98锛堝钩绋虫敹鏁涳級 |

### 寰皟鍓嶅悗瀵规瘮锛堣瑙?`eval/reports/compare_report.md`锛?

| 鐢ㄤ緥 | 鍩哄骇 `deepseek-r1:1.5b` | 寰皟鍚?`deepseek-r1-roleplay` |
|---|---|---|
| 鍑夊鏄ユ棩 | 鍑烘垙锛屾硾娉涜€岃皥 | 浠ャ€屻€嶅彛鍚荤殑瑙掕壊鍙拌瘝锛岀鍚堜汉璁?|
| 寰″潅缇庣惔 | 鍍忕櫨绉戦棶绛旓紝鏃犺鑹叉劅 | 瑙掕壊鍙ｅ惢鍥炵瓟锛屽甫銆屻€?|
| 闊﹀皬瀹?| 鐩存帴鎷掔瓟銆屾垜杩樻病瀛︿細鍥炵瓟銆?| 浠ラ煢灏忓疂璇皵鍥炴€硷紝绗﹀悎浜鸿 |
| 鏁板鎺ㄧ悊锛堥槻閬楀繕锛?| 姝ｇ‘瑙ｅ嚭 x=-2 | 姝ｇ‘瑙ｅ嚭 x=-2 鉁?鏈仐蹇?|
| 鏃ュ父瀵硅瘽 | 姝ｅ父 | 鐣ラ€€鍖栵紙閲嶅鍫嗗彔锛夛紝灏忔ā鍨嬪皬鏁版嵁閲忕殑棰勬湡鍐呬唬浠?|

### EvalScope 鍩哄噯璇勬祴锛圖ashboard锛?

![EvalScope 璇勬祴鎴愮哗浠〃鐩榏(eval/reports/dashboard.png)

| 鏁版嵁闆?| 鍩哄骇 `deepseek-r1:1.5b` | 寰皟鍚?`deepseek-r1-roleplay` |
|---|---|---|
| GSM8K锛堟暟瀛︽帹鐞嗭級 | 20% | **60%** |
| ARC锛堝父璇嗘帹鐞嗭級 | 30% | 35% |

> 鐢?EvalScope锛坥penai_api 妯″紡锛夌洿杩?Ollama 鏈嶅姟璇勬祴锛屾瘡涓暟鎹泦 10 鏉℃牱鏈紙limit=10锛宮ax_tokens=1024锛夛紝
> 鍒嗘暟涓轰富鎸囨爣 Accuracy銆傚師濮嬫姤鍛婂湪 `eval/evalscope/*/reports/`锛屼华琛ㄧ洏鐢?`eval/make_dashboard.py` 鐢熸垚銆?

### 鍘嬫祴锛堣瑙?`eval/reports/benchmark.json`锛?

| 鎸囨爣 | 鏁板€?|
|---|---|
| 璇锋眰鏁?| 20锛坄num_predict=256`锛?|
| 鎴愬姛鐜?| **20/20 = 100%** |
| 鍚炲悙 | **57.7 tokens/s**锛圧TX 4060 Laptop锛孫llama 鎺ㄧ悊锛?|

## 浣跨敤寰皟鍚庣殑妯″瀷

```powershell
ollama run deepseek-r1-roleplay
```

鎴栬蛋 OpenAI 鍏煎鎺ュ彛锛坄http://localhost:11434/v1`锛夛紝鎶?`model` 璁句负 `deepseek-r1-roleplay`銆?

> 浠撳簱鍐呴檮甯?LoRA adapter锛坄weights/lora_adapter`锛?1MB锛夛紝鍙敤 `train/merge.py` 涓庡熀搴у悎骞跺悗澶嶇幇锛?
> 鍚堝苟鍚庣殑 fp16 鏉冮噸锛?.4GB锛変笌 GGUF锛?.55GB锛変綋绉繃澶ф湭鍏ュ簱锛屾寜涓婃柟銆屽鐜版楠ゃ€嶉噸鏂扮敓鎴愬嵆鍙€?


## 鏁版嵁闆?

- [ChatHaruhi-54K-Role-Playing-Dialogue](https://huggingface.co/datasets/silk-road/ChatHaruhi-54K-Role-Playing-Dialogue)锛?4K 瑙掕壊鎵紨瀵硅瘽锛堝噳瀹槬鏃ャ€佸尽鍧傜編鐞寸瓑 32 涓粡鍏歌鑹诧級锛孋C-BY-4.0
- [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)锛氶€氱敤瀵硅瘽锛岀敤浜庨槻姝㈢伨闅炬€ч仐蹇?

