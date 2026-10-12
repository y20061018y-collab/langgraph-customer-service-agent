"""
LLM 只在这里初始化一次，其他地方都不应该直接 new 一个 ChatOpenAI。
这样以后想从 OpenAI 切换到智谱 GLM / DeepSeek 等任何 OpenAI 兼容接口，
只需要改 .env 里的 BASE_URL 和 MODEL_NAME，不用动任何业务代码。

为什么用 ChatOpenAI 而不是各家专属 SDK：
绝大多数国产模型（智谱GLM、DeepSeek、Moonshot等）都提供了 OpenAI 兼容的
/chat/completions 接口，所以复用 langchain-openai 的 ChatOpenAI 客户端，
换 base_url 就能接入，不需要为每个厂商单独写一套 LLM 封装。
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm():
    """
    从环境变量读取配置，返回一个已绑定好的 ChatOpenAI 实例。
    .env 里需要配置：
        LLM_API_KEY=你的key
        LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/   # 智谱GLM举例，换厂商就换这行
        LLM_MODEL_NAME=glm-4-plus                              # 换成你要用的模型名
    """
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        raise RuntimeError(
            "没有找到 LLM_API_KEY，请复制 .env.example 为 .env 并填入你的真实 key"
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,  # 为 None 时走 OpenAI 官方地址
        model=model_name,
        temperature=0,
    )
