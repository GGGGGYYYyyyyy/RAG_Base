import requests
import json
import logging
import config.config as app_config # 确保导入配置

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, api_url: str, model_name: str, system_prompt: str = None):
        self.api_url = api_url
        self.model_name = model_name
        # 如果 system_prompt 为 None，则尝试从 config 加载，否则使用提供的
        self.system_prompt = system_prompt if system_prompt is not None else getattr(app_config, 'DEFAULT_SYSTEM_PROMPT', self._get_fallback_system_prompt())

    def _get_fallback_system_prompt(self) -> str:
        """如果配置中也没有，提供一个最终的后备系统提示词"""
        logger.warning("DEFAULT_SYSTEM_PROMPT 未在配置中找到，使用内部后备提示词。")

        return (
            "...."
        )

    def _format_contexts_for_prompt(self, contexts: list, max_contexts_for_llm: int) -> str:
        # (此方法保持不变，但确保 max_contexts_for_llm 从配置或参数正确传入)
        actual_max_contexts = max_contexts_for_llm if max_contexts_for_llm is not None else getattr(app_config, 'MAX_CONTEXTS_FOR_LLM_PROMPT', 5)
        context_texts = []
        if not contexts:
            logger.warning("没有提供上下文给 LLM。")
            return "（上下文中未找到相关信息）"

        for i, res_item in enumerate(contexts[:actual_max_contexts]): # 使用 actual_max_contexts
            content_details = res_item.get("content_details", {})
            doc_content = content_details.get("content", "内容缺失")
            doc_title = content_details.get("doc_title", "未知文档")
            chunk_id = res_item.get("chunk_id", f"未知块_{i+1}")

            if not doc_content or not doc_content.strip():
                doc_content = "（此块内容为空或无法加载）"
                logger.warning(f"Context item {i} (Chunk ID: {chunk_id}) has empty content.")

            context_texts.append(
                f"--- 参考资料 {i+1} ---\n"
                f"来源文档: {doc_title}\n"
                f"片段ID: {chunk_id}\n"
                f"内容: {doc_content.strip()}\n"
                f"--- 参考资料 {i+1} 结束 ---\n"
            )
        full_context_str = "\n".join(context_texts)
        return full_context_str

    def _prepare_payload(self, query: str, contexts: list, max_tokens: int,
                         temperature: float, top_p: float = None,
                         presence_penalty: float = None, frequency_penalty: float = None,
                         max_contexts_for_llm: int = None, stream: bool = False) -> dict:
        """准备发送给LLM API的请求体"""
        
        # 从配置或参数获取默认值
        current_max_tokens = max_tokens if max_tokens is not None else getattr(app_config, 'LLM_DEFAULT_MAX_TOKENS', 4096)
        current_temperature = temperature if temperature is not None else getattr(app_config, 'LLM_DEFAULT_TEMPERATURE', 0.1)
        current_max_contexts = max_contexts_for_llm if max_contexts_for_llm is not None else getattr(app_config, 'MAX_CONTEXTS_FOR_LLM_PROMPT', 5)
        current_stream = stream if stream is not None else getattr(app_config, 'LLM_STREAM_OUTPUT', False)


        formatted_context = self._format_contexts_for_prompt(contexts, current_max_contexts)
        user_message_content = (
            f"请分析以下参考资料，并针对用户提出的问题，提炼出完整详细且关键信息点进行回答。\n\n"
            f"--- 参考资料开始 ---\n"
            f"{formatted_context}\n"
            f"--- 参考资料结束 ---\n\n"
            f"问题: {query}\n\n"
            f"请基于参考资料，以清晰、完整、详细的方式总结并回答，确保所有论点均有资料支持。"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message_content}
            ],
            "max_tokens": current_max_tokens,
            "temperature": current_temperature,
            "stream": current_stream
        }

        # 获取可选参数的配置默认值
        current_top_p = top_p if top_p is not None else getattr(app_config, 'LLM_DEFAULT_TOP_P', None)
        current_presence_penalty = presence_penalty if presence_penalty is not None else getattr(app_config, 'LLM_DEFAULT_PRESENCE_PENALTY', None)
        current_frequency_penalty = frequency_penalty if frequency_penalty is not None else getattr(app_config, 'LLM_DEFAULT_FREQUENCY_PENALTY', None)

        if current_top_p is not None: payload["top_p"] = current_top_p
        if current_presence_penalty is not None: payload["presence_penalty"] = current_presence_penalty
        if current_frequency_penalty is not None: payload["frequency_penalty"] = current_frequency_penalty
        
        return payload

    def generate_answer_stream(self, query: str, contexts: list,
                               max_tokens: int = None, temperature: float = None,
                               top_p: float = None, presence_penalty: float = None,
                               frequency_penalty: float = None,
                               max_contexts_for_llm: int = None):
        """
        流式生成答案并打印到控制台。
        """
        payload = self._prepare_payload(
            query, contexts, max_tokens, temperature, top_p,
            presence_penalty, frequency_penalty, max_contexts_for_llm, stream=True
        )

        logger.info(f"发送流式请求到 LLM API: {self.api_url} for query: '{query[:50]}...'")
        full_response_content = []

        try:
            with requests.post(self.api_url, json=payload, stream=True, timeout=180) as response:
                response.raise_for_status()
                if response.encoding is None: # 确保正确解码
                    response.encoding = 'utf-8'

                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith('data: '):
                        line_data = line[len('data: '):].strip()
                        if line_data == "[DONE]":
                            print() # 换行结束
                            logger.info("LLM 流式输出结束。")
                            break
                        try:
                            chunk = json.loads(line_data)
                            if chunk.get("choices") and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content_piece = delta.get("content")
                                if content_piece:
                                    print(content_piece, end="", flush=True)
                                    full_response_content.append(content_piece)
                        except json.JSONDecodeError:
                            logger.warning(f"无法解码流中的 JSON 数据块: {line_data}")
                            continue # 跳过无法解析的行
            return "".join(full_response_content) # 返回完整拼接的响应

        except requests.exceptions.Timeout:
            logger.error(f"调用 LLM API (流式) 超时: {self.api_url}")
            print("\n错误：请求大语言模型超时，请稍后再试。")
            return "错误：请求大语言模型超时，请稍后再试。"
        except requests.exceptions.RequestException as e:
            logger.error(f"调用 LLM API (流式) 时发生网络或HTTP错误: {e}")
            print(f"\n错误：无法连接到大语言模型服务 ({e})。")
            return f"错误：无法连接到大语言模型服务 ({e})。"
        except Exception as e:
            logger.error(f"调用 LLM (流式) 时发生未知错误: {e}", exc_info=True)
            print("\n错误：生成回答时发生未知内部错误。")
            return "错误：生成回答时发生未知内部错误。"


    def generate_answer_non_stream(self, query: str, contexts: list,
                                   max_tokens: int = None, temperature: float = None,
                                   top_p: float = None, presence_penalty: float = None,
                                   frequency_penalty: float = None,
                                   max_contexts_for_llm: int = None) -> str:
        """
        非流式生成答案。
        """
        payload = self._prepare_payload(
            query, contexts, max_tokens, temperature, top_p,
            presence_penalty, frequency_penalty, max_contexts_for_llm, stream=False
        )

        logger.info(f"发送非流式请求到 LLM API: {self.api_url} for query: '{query[:50]}...'")
        # logger.debug(f"LLM Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            llm_response_data = response.json()

            if llm_response_data.get("choices") and len(llm_response_data["choices"]) > 0:
                message_content = llm_response_data["choices"][0].get("message", {}).get("content")
                if message_content:
                    logger.info("LLM (非流式) 成功生成回答。")
                    return message_content.strip()
                else:
                    logger.error(f"LLM (非流式) 响应中的 'content' 字段为空或缺失: {llm_response_data}")
                    return "错误：LLM 返回了空的回答内容。"
            else:
                logger.error(f"LLM (非流式) 响应格式不符合预期: {llm_response_data}")
                return "错误：LLM 响应格式不符合预期。"
        # ... (错误处理与流式版本类似) ...
        except requests.exceptions.Timeout:
            logger.error(f"调用 LLM API (非流式) 超时: {self.api_url}")
            return "错误：请求大语言模型超时，请稍后再试。"
        except requests.exceptions.RequestException as e:
            logger.error(f"调用 LLM API (非流式) 时发生网络或HTTP错误: {e}")
            return f"错误：无法连接到大语言模型服务 ({e})。"
        except json.JSONDecodeError as e:
            logger.error(f"解码 LLM API (非流式) 的 JSON 响应失败: {e}. 响应文本: {response.text if 'response' in locals() else 'N/A'}")
            return "错误：无法解析大语言模型的响应。"
        except Exception as e:
            logger.error(f"调用 LLM (非流式) 时发生未知错误: {e}", exc_info=True)
            return "错误：生成回答时发生未知内部错误。"
        
    async def generate_answer_stream_async(self, query: str, contexts: list,
                                     max_tokens: int = None, temperature: float = None,
                                     top_p: float = None, presence_penalty: float = None,
                                     frequency_penalty: float = None,
                                     max_contexts_for_llm: int = None): # 修改为 async def
        """
        异步流式生成答案，并 yield 每个数据块。
        """
        payload = self._prepare_payload(
            query, contexts, max_tokens, temperature, top_p,
            presence_penalty, frequency_penalty, max_contexts_for_llm, stream=True
        )

        logger.info(f"发送异步流式请求到 LLM API: {self.api_url} for query: '{query[:50]}...'")

        try:
            # 使用 httpx 进行异步请求
            import httpx
            async with httpx.AsyncClient(timeout=180.0) as client: # 使用 httpx.AsyncClient
                async with client.stream("POST", self.api_url, json=payload) as response:
                    response.raise_for_status()
                    # if response.encoding is None: # httpx 会自动处理编码
                    #     response.encoding = 'utf-8'

                    async for line in response.aiter_lines(): # 使用 aiter_lines
                        if line and line.startswith('data: '):
                            line_data = line[len('data: '):].strip()
                            if line_data == "[DONE]":
                                logger.info("LLM 流式输出结束。")
                                break # 正常结束
                            try:
                                chunk = json.loads(line_data)
                                if chunk.get("choices") and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    content_piece = delta.get("content")
                                    if content_piece:
                                        yield content_piece # yield 数据块
                            except json.JSONDecodeError:
                                logger.warning(f"无法解码流中的 JSON 数据块: {line_data}")
                                continue
        except httpx.TimeoutException: # 捕获 httpx 的超时
            logger.error(f"调用 LLM API (异步流式) 超时: {self.api_url}")
            yield "错误：请求大语言模型超时，请稍后再试。" # 返回错误信息
        except httpx.RequestError as e: # 捕获 httpx 的请求错误
            logger.error(f"调用 LLM API (异步流式) 时发生网络或HTTP错误: {e}")
            yield f"错误：无法连接到大语言模型服务 ({e})。"
        except Exception as e:
            logger.error(f"调用 LLM (异步流式) 时发生未知错误: {e}", exc_info=True)
            yield "错误：生成回答时发生未知内部错误。"