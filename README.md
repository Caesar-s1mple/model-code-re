# 主流模型结构代码

为什么有这个repo？写这个主要因为实在受不了网上资源的不可靠性和匮乏性了。你说一个Transformer都出了几年了，网上搜一下：“Transformer模型代码”，真的找不到一个写的特别规范特别好的代码，每次看到写的一坨屎的代码评论区还一堆点赞的真的感慨。于是乎我打算自己构建整理这个包含大部分主流模型结构代码的仓库，提供一个规范的模型结构学习代码，顺带巩固自己的知识。顺带写了众多其他代码供学习。

能学到什么代码：

    模型代码！推理代码！项目组织结构参考！
    推理加速：量化！投机解码！torch.compile！

包含模型：

    Transformer
    BERT
    VisionTransformer               不包含推理代码
    Llama-3
    GPT-NEOX

预计在不久加入：

    1. 测试推理加速的Benchmark
    2. 训练代码：SFT、LoRA。
    3. 更多的模型
    

## 1. 项目结构说明

项目代码目录管理非常重要，一个清晰、符合大众认知的目录层级管理能够让使用者更快速、更有意愿去上手你的代码。

```
model-code-re/
├── models/
│   ├── config/                 模型的参数文件~
│   │   ├── transformer.json
│   │   ├── llama-3-7b_compile.json
│   │   └── ...
│   ├── utils/                  全是工具代码~
│   │   ├── __init__.py
│   │   ├── quantize.py         模型量化代码~
│   │   ├── tokenizer.py        顾名思义~
│   │   └── utils.py            一些泛用函数，如sample、set_seed~
│   ├── structure_images/       供学习模型结构的参考图~
│   │   ├── transformer.jpg
│   │   ├── llama.jpg
│   │   └── ...
│   ├── __init__.py
│   ├── transformer.py          模型代码！💡💡💡
│   ├── llama_compile.py        模型代码！💡💡💡
│   └── ...
├── runs_ar/                    各模型常规自回归生成代码~
│   ├── transformer.py
│   ├── llama_compile.py
│   └── ...
├── runs_ss/                    各模型投机采样生成代码~
│   ├── transformer.py
│   ├── llama_compile.py
│   └── ...
├── checkpoints/                从huggingface🤗上下载的repo~
│   ├── Meta-Llama-3-8B/
│   │   └── ...
│   └── ...
├── convert_hf_checkpoint.py    把huggingface模型权重转换为本repo代码能加载的
├── show_structure.py           展示模型结构图~
├── app.py                      启动一个FastAPI服务器以接口形式调用~
├── run.py                      常规的模型推理~
└── README.md                   it's me~😊
```

## 2. 快速上手

- ### 2.1 想看模型代码怎么写？

    直接看models/文件夹下的模型代码即可，全都是独立的代码。

- ### 2.2 想看模型推理怎么写？

    普通的自回归生成直接看runs_ar/文件夹下的代码即可。

    使用了投机采样的直接看runs_ss/文件夹下的代码即可。

- ### 2.3 想看一些杂七杂八的工具（量化、tokenizer等）怎么写？

    欢迎看models/utils/文件夹下的代码。

    注：quantize.py和tokenizer.py并不是我自己写的，PyTorch官方写的！觉得写的一坨（有些地方真的一坨）别赖我。

- ### 2.4 我就是想跑着玩！（反正就是想跑起来）

    ok，最复杂的需求来了。首先观察一下这个repo，应有尽有了吧，足够清晰了吧。还缺什么？模型权重对吧。

    首先需要去[HuggingFace](https://huggingface.co/)上下载对应模型的权重到checkpoints/文件夹下，比如[Llama-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B)。

    但是下载到的模型权重无法直接用，因为其模型属性名（也就是参数的名字）和我定义的名字对不上，因此需要用到模型转换脚本convert_hf_checkpoint.py。
    
    在这里我也许可以给你提供每个模型的权重映射文件，然后写成一个直接设置参数就可以调用进行转换的脚本。但我不！我要教你怎么做。

    1. 首先需要知道从HuggingFace上下载到的模型原始权重是什么东西，基础知识，是个有序字典（正常来说是OrderedDict，但下载到的其实是个有序的dict，无所谓，不重要）。
  
    2. 自己写一个类似以下的脚本看下下载到的模型结构和参数名：
       ```python
       import torch 
       
       weights = torch.load('./checkpoints/Meta-Llama-3-8B/original/consolidated.00.pth')
       
       for name, param in weights.items():
           print(f'{name}: {param.shape}')
       ```
       再写一个类似以下的脚本看下我们写的模型结构和参数名：
       ```python
       import torch
       from models import LLaMA
       from models.utils import Config
           
       with torch.device('meta'):
           model = LLaMA(Config('./models/config/llama-3-8b_compile.json'))
           
       for name, param in model.named_parameters():
           print(f'{name}: {param.shape}')
       ```
       同时运行这两个脚本对比结果，你可以很轻易的写出一个像convert_hf_checkpoint.py中那样的weight_map。
       
       将weight_map换成你写的，指定原始HuggingFace的repo路径，运行！

    3. 于是在repo的路径下的convert文件夹下得到了转换好的模型权重model.pth，顺带把tokenizer.model也拷贝了过来。

    4. 至此基本大功告成。
### 防喷说明

如果你觉得我模型代码的命名、写法、逻辑等有问题！请先思考一下是否我这样写更好（先尝试说服一下自己）。

如果思考后还觉得有问题，欢迎issue里指出，或者在尽量不改变代码风格的前提下直接扔我个Pull Requests~

Q&A：
1. 为什么不把普通的自回归解码和投机解码整合成一个运行脚本，而要分开到两个文件夹下？

    ***因为将两部分逻辑合在一起会多出很多繁琐的判断逻辑，导致一个本来简单的代码变得看起来非常复杂。把两部分分开更容易达到清晰学习的目的。***
2. 有些模型/运行脚本的名字后面带个compile是什么意思？

    ***具体可以参考这篇[PyTorch官方blog](https://pytorch.org/blog/accelerating-generative-ai-2/)，是一种高效推理的方案。***

3. 我想转换模型权重，但是发现官方权重和我的对不上？

    ***这可能发生于Attention的Wq、Wk、Wv矩阵，一般情况下是分为三个linear的，但是有的模型会将三个矩阵合起来变成linear_qkv，效果是不变的。对于这种情况模型代码里已经写好处理了，你只需要将他们分别映射为linear_q、linear_w、linear_v。如果还是不能理解，看一眼模型代码里的load_hook就知道了。***
