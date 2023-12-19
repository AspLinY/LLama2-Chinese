import argparse  # 解析命令行参数
import gc  # 内存管理
import math  # 数学运算支持
import os  # 与操作系统交互
import time  # 时间相关操作

import datetime  # 处理日期时间
import json  # 处理 JSON 数据
import torch  # 机器学习库
import torch.distributed as dist  # PyTorch 的分布式计算模块
import uvicorn  # uvicorn用于托管 FastAPI 应用
from fastapi import FastAPI, Request  # 创建 web 应用和处理请求
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # 再次从 transformers 导入模型和Tokenizer
from transformers import AutoTokenizer, AutoModel  # 从 transformers 库导入自动选择合适的Tokenizer和模型

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)  # 模型的本地路径
parser.add_argument('--gpus', default="0", type=str)  # 使用的显卡编号
parser.add_argument('--infer_dtype', default="int8", choices=["int4", "int8", "float16"], required=False,
                    type=str)  # 模型加载后的参数数据类型
parser.add_argument('--model_source', default="llama2_chinese", choices=["llama2_chinese", "llama2_meta"],  # 模型的源
                    required=False, type=str)

args = parser.parse_args()  # 解析命令行输入的参数
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # 设置环境变量以控制 PyTorch 使用的 GPU

local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 获取和设置分布式计算的相关变量/如果环境变量中没有设置 LOCAL_RANK，它将默认返回 "0"
# LOCAL_RANK 通常用来标识当前进程所在的 GPU 编号

world_size = torch.cuda.device_count()  # 获取当前环境中可用的 GPU 数量

rank = local_rank

app = FastAPI()  # 创建一个 FastAPI 应用实例


# ========构建用于LLaMA2-Chinese模型的输入提示（prompt）========##
def get_prompt_llama2chinese(chat_history, system_prompt: str) -> str:
    # 它接受两个参数：chat_history（聊天历史记录）和 system_prompt（系统提示），并返回一个字符串

    prompt = ''  # 初始化一个空字符串，用于构建最终的输入提示

    for input_text_one in chat_history:  # 遍历 chat_history 中的每一条记录
        prompt += "<s>" + input_text_one['role'] + ": " + input_text_one['content'].strip() + "\n</s>"
        # 对于每条聊天记录，将角色（例如 "Human" 或 "Assistant"）和内容添加到提示字符串中，并用特定的标签 <s> 和 </s> 包裹每条消息

    if chat_history[-1]['role'] == 'Human':  # 检查最后一条消息是否来自人类用户
        prompt += "<s>Assistant: "  # 如果是，那么下一条消息应该来自助手，因此添加助手的提示
    else:
        prompt += "<s>Human: "  # 否则，下一条消息应该来自人类，添加相应的提示。
    prompt = prompt[-2048:]  # 输入长度限制
    if len(system_prompt) > 0:  # 检查是否有系统提示
        prompt = '<s>System: ' + system_prompt.strip() + '\n</s>' + prompt  # 如果有，将系统提示添加到最终提示的前面，并用 <s> 和 </s> 标签包裹

    return prompt  # 返回构建好的提示字符串


###############################################

# ========构建用于模型的输入提示（prompt）========##
def get_prompt(chat_history, system_prompt: str):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    sep = " "
    sep2 = " </s><s>"
    stop_token_ids = [2]
    system_template = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    roles = ("[INST]", "[/INST]")
    seps = [sep, sep2]
    if system_prompt.strip() != "":
        ret = system_template
    else:
        ret = "[INST] "
    for i, chat in enumerate(chat_history):
        message = chat["content"]
        role = chat["role"]
        if message:
            if i == 0:
                ret += message + " "
            else:
                if role == "Human":
                    ret += "[INST]" + " " + message + seps[i % 2]
                else:
                    ret += "[/INST]" + " " + message + seps[i % 2]
        else:
            if role == "Human":
                ret += "[INST]"
            else:
                ret += "[/INST]"
    print("prompt:{}".format(ret))
    return ret


@app.post("/generate")  # 指定该函数处理对 /generate 路径的 POST 请求
async def create_item(request: Request):  # 异步函数定义，接收一个请求对象
    global model, tokenizer

    # 解析 JSON 请求体
    json_post_raw = await request.json()  # 异步获取 JSON 数据
    json_post = json.dumps(json_post_raw)  # 将 JSON 对象转换为字符串
    json_post_list = json.loads(json_post)  # 再次将字符串转换回 JSON 对象

    # 提取数据
    history = json_post_list.get('history')  # 获取聊天历史
    # 获取其他模型生成所需的参数
    system_prompt = json_post_list.get('system_prompt')
    max_new_tokens = json_post_list.get('max_new_tokens')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    # **************************************[构建模型输入]**************************************#
    prompt = get_prompt_llama2chinese(history, system_prompt)  # 使用history, system_prompt构建模型的prompt
    inputs = tokenizer([prompt], return_tensors='pt').to("cuda")  # 使用分词器处理输入提示，并转换为 PyTorch 张量，准备在 CUDA 上运行
    # ****************************************************************************************#
    generate_kwargs = dict(
        inputs,
        # streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=50,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=1.2,
        max_length=2048,
    )  # 构建包含所有生成参数的字典
    generate_ids = model.generate(**generate_kwargs)  # 使用这些参数生成响应

    generate_ids = [item[len(inputs[0]):-1] for item in generate_ids]  # 处理生成的响应 ID，去除不必要的部分

    bot_message = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # 将生成的 ID 解码为可读文本

    if 'Human:' in bot_message:
        bot_message = bot_message.split('Human:')[0]

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": bot_message,
        "status": 200,
        "time": time
    }  # 构建响应对象
    return answer


def get_world_size() -> int:  # 用于获取分布式环境中的“world size”，即参与计算的总节点数
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def print_rank0(*msg):  # 用于在多节点环境中仅在“rank 0”节点打印消息，通常用于避免在每个节点上重复打印相同的信息
    if rank != 0:
        return
    print(*msg)


if __name__ == '__main__':
    dtype = torch.float16
    kwargs = dict(
        device_map="auto",
    )
    print("get_world_size:{}".format(get_world_size()))

    infer_dtype = args.infer_dtype
    if infer_dtype not in ["int4", "int8", "float16"]:
        raise ValueError("infer_dtype must one of int4, int8 or float16")

    if get_world_size() > 1:
        kwargs["device_map"] = "balanced_low_0"

    if infer_dtype == "int8":
        print_rank0("Using `load_in_8bit=True` to use quanitized model")
        kwargs["load_in_8bit"] = True
    else:
        kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if infer_dtype in ["int8", "float16"]:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    elif infer_dtype == "int4":
        from auto_gptq import AutoGPTQForCausalLM, get_gptq_peft_model

        model = AutoGPTQForCausalLM.from_quantized(
            args.model_path, device="cuda:0",
            use_triton=False,
            low_cpu_mem_usage=True,
            # inject_fused_attention=False,
            # inject_fused_mlp=False
        )

    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8001, workers=1)
